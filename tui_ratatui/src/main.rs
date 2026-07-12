//! GT Diagnostic Harness — ratatui frontend.
//!
//! Diagnosis engine stays in Python:
//!   `python app.py --json-once <csv> --mode …`
//!
//! Env:
//!   GT_HARNESS_ROOT   — project root
//!   GT_HARNESS_PYTHON — python executable
//!   GT_NO_DOWNLOAD=1  — offline engine
//!   GT_DEFAULT_CSV    — optional default path

use anyhow::{anyhow, Context, Result};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};
use serde::Deserialize;
use std::{
    env, io,
    path::{Path, PathBuf},
    process::Command,
    sync::mpsc,
    thread,
    time::Duration,
};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

// ── Bridge JSON ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Deserialize)]
struct Severity {
    severity: Option<f64>,
    level: Option<String>,
    #[allow(dead_code)]
    label: Option<String>,
    #[allow(dead_code)]
    top_channel: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ProofChannel {
    name: String,
    score: f64,
    #[serde(default)]
    #[allow(dead_code)]
    flag_count: u32,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct BridgeResult {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    mode: Option<String>,
    #[serde(default)]
    anomaly_mode: Option<String>,
    #[serde(default)]
    severity: Option<Severity>,
    #[serde(default)]
    final_report: Option<String>,
    #[serde(default)]
    display_markdown: Option<String>,
    #[serde(default)]
    evidence_ascii: Option<String>,
    #[serde(default)]
    proof_channels: Vec<ProofChannel>,
    #[serde(default)]
    elapsed_s: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    csv_path: Option<String>,
}

// ── Paths ──────────────────────────────────────────────────────────────────

fn find_project_root() -> PathBuf {
    if let Ok(r) = env::var("GT_HARNESS_ROOT") {
        let p = PathBuf::from(r.trim());
        if p.join("app.py").is_file() {
            return p;
        }
    }
    if let Ok(exe) = env::current_exe() {
        for anc in exe.ancestors() {
            if anc.join("app.py").is_file() && anc.join("samples").is_dir() {
                return anc.to_path_buf();
            }
        }
    }
    if let Ok(cwd) = env::current_dir() {
        for anc in cwd.ancestors() {
            if anc.join("app.py").is_file() {
                return anc.to_path_buf();
            }
        }
    }
    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn normalize_path_text(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    if text.is_empty() {
        return text;
    }
    if let Some(line) = text.lines().map(str::trim).find(|l| !l.is_empty()) {
        text = line.to_string();
    }
    let b = text.as_bytes();
    if b.len() >= 2
        && ((b[0] == b'"' && b[b.len() - 1] == b'"') || (b[0] == b'\'' && b[b.len() - 1] == b'\''))
    {
        text = text[1..text.len() - 1].trim().to_string();
    }
    let lower = text.to_ascii_lowercase();
    if lower.starts_with("file:") {
        let rest = text
            .trim_start_matches("file:")
            .trim_start_matches('/')
            .trim_start_matches('\\');
        if rest.len() >= 2 && rest.as_bytes()[1] == b':' {
            text = rest.to_string();
        } else {
            text = rest.to_string();
        }
    }
    text
}

fn resolve_csv_path(raw: &str, project_root: &Path) -> Option<PathBuf> {
    let text = normalize_path_text(raw);
    if text.is_empty() {
        return None;
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    let direct = PathBuf::from(&text);
    candidates.push(direct.clone());
    if direct.is_relative() {
        candidates.push(project_root.join(&direct));
        if let Some(name) = direct.file_name() {
            candidates.push(project_root.join("samples").join(name));
            let samples = project_root.join("samples");
            if samples.is_dir() {
                if let Ok(rd) = std::fs::read_dir(&samples) {
                    for ent in rd.flatten() {
                        let p = ent.path();
                        if p.is_dir() {
                            candidates.push(p.join(name));
                        }
                    }
                }
            }
        }
    }
    if text.contains('/') {
        candidates.push(PathBuf::from(text.replace('/', "\\")));
        candidates.push(project_root.join(text.replace('/', "\\")));
    }
    for c in candidates {
        if c.is_file() {
            return Some(c.canonicalize().unwrap_or(c));
        }
    }
    None
}

fn list_sample_csvs(project_root: &Path) -> Vec<(String, PathBuf)> {
    let mut out: Vec<(String, PathBuf)> = Vec::new();
    let samples = project_root.join("samples");
    let demo = samples.join("gt_sensors_demo.csv");
    if demo.is_file() {
        out.push(("Demo · gt_sensors_demo.csv".into(), demo));
    }
    if samples.is_dir() {
        if let Ok(rd) = std::fs::read_dir(&samples) {
            let mut dirs: Vec<_> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
            dirs.sort();
            for dir in dirs {
                let folder = dir
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "sample".into());
                if let Ok(files) = std::fs::read_dir(&dir) {
                    let mut csvs: Vec<_> = files
                        .flatten()
                        .map(|e| e.path())
                        .filter(|p| {
                            p.extension()
                                .and_then(|e| e.to_str())
                                .map(|e| e.eq_ignore_ascii_case("csv"))
                                .unwrap_or(false)
                        })
                        .collect();
                    csvs.sort();
                    for csv in csvs {
                        let name = csv
                            .file_name()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default();
                        out.push((format!("{folder} · {name}"), csv));
                    }
                }
            }
        }
    }
    out
}

fn default_demo_csv(project_root: &Path) -> String {
    if let Ok(p) = env::var("GT_DEFAULT_CSV") {
        if let Some(resolved) = resolve_csv_path(&p, project_root) {
            return resolved.display().to_string();
        }
    }
    let demo = project_root.join("samples").join("gt_sensors_demo.csv");
    if demo.is_file() {
        return demo.canonicalize().unwrap_or(demo).display().to_string();
    }
    demo.display().to_string()
}

// ── App state ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    Samples,
    PathEdit,
    ContextEdit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Alerts,
    Trips,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Alerts => "Alerts",
            Mode::Trips => "Trips/Event",
        }
    }
    fn toggle(self) -> Self {
        match self {
            Mode::Alerts => Mode::Trips,
            Mode::Trips => Mode::Alerts,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pane {
    Plots,
    Report,
}

struct SampleItem {
    label: String,
    path: PathBuf,
}

struct App {
    csv_path: String,
    context: String,
    mode: Mode,
    focus: Focus,
    pane: Pane,
    status: String,
    busy: bool,
    last: Option<BridgeResult>,
    scroll_plots: u16,
    scroll_report: u16,
    project_root: PathBuf,
    python: String,
    samples: Vec<SampleItem>,
    sample_state: ListState,
    rx: Option<mpsc::Receiver<Result<BridgeResult>>>,
    log_lines: Vec<String>,
}

impl App {
    fn new() -> Self {
        let root = find_project_root();
        env::set_var("GT_HARNESS_ROOT", &root);
        let python = env::var("GT_HARNESS_PYTHON").unwrap_or_else(|_| {
            // Prefer `py -3` is not a single path; keep python/python3
            if which_exists("python") {
                "python".into()
            } else {
                "python3".into()
            }
        });
        let samples: Vec<SampleItem> = list_sample_csvs(&root)
            .into_iter()
            .map(|(label, path)| SampleItem { label, path })
            .collect();
        let mut sample_state = ListState::default();
        if !samples.is_empty() {
            sample_state.select(Some(0));
        }
        let csv = if let Some(first) = samples.first() {
            first
                .path
                .canonicalize()
                .unwrap_or_else(|_| first.path.clone())
                .display()
                .to_string()
        } else {
            default_demo_csv(&root)
        };
        let ok = resolve_csv_path(&csv, &root).is_some();
        let mut log_lines = vec![
            format!("Project root: {}", root.display()),
            format!("Python: {python}"),
            format!("Samples found: {}", samples.len()),
        ];
        if !ok {
            log_lines.push("WARNING: default CSV not found — pick a sample or edit path (e)".into());
        }
        Self {
            csv_path: csv,
            context: String::new(),
            mode: Mode::Alerts,
            focus: Focus::Samples, // navigation — keys do NOT corrupt the path
            pane: Pane::Plots,
            status: if ok {
                "Ready · ↑↓ pick sample · Enter select · r/F5 Run · m Mode · 1/2 panes · q Quit"
                    .into()
            } else {
                "No default CSV · pick a sample or press e to edit path".into()
            },
            busy: false,
            last: None,
            scroll_plots: 0,
            scroll_report: 0,
            project_root: root,
            python,
            samples,
            sample_state,
            rx: None,
            log_lines,
        }
    }

    fn push_log(&mut self, line: impl Into<String>) {
        self.log_lines.push(line.into());
        if self.log_lines.len() > 40 {
            let n = self.log_lines.len() - 40;
            self.log_lines.drain(0..n);
        }
    }

    fn selected_sample_index(&self) -> Option<usize> {
        self.sample_state.selected()
    }

    fn apply_selected_sample(&mut self) {
        if let Some(i) = self.selected_sample_index() {
            if let Some(item) = self.samples.get(i) {
                let p = item
                    .path
                    .canonicalize()
                    .unwrap_or_else(|_| item.path.clone());
                self.csv_path = p.display().to_string();
                self.status = format!("Selected: {}", item.label);
                self.push_log(format!("CSV → {}", self.csv_path));
            }
        }
    }

    fn chips_line(&self) -> String {
        let sev = self
            .last
            .as_ref()
            .and_then(|r| r.severity.as_ref())
            .map(|s| {
                format!(
                    "{} ({})",
                    s.level.clone().unwrap_or_else(|| "—".into()),
                    s.severity
                        .map(|v| format!("{v:.2}"))
                        .unwrap_or_else(|| "—".into())
                )
            })
            .unwrap_or_else(|| "—".into());
        let eng = self
            .last
            .as_ref()
            .and_then(|r| r.anomaly_mode.clone())
            .unwrap_or_else(|| "—".into());
        let busy = if self.busy { "RUNNING" } else { "idle" };
        format!(
            "ratatui  ·  {}  ·  SEV {}  ·  ENG {}  ·  {}",
            self.mode.label(),
            sev,
            eng,
            busy
        )
    }

    fn start_diagnosis(&mut self) {
        if self.busy {
            self.status = "Already running…".into();
            return;
        }
        // Leave edit mode so keys don't corrupt path mid-run
        self.focus = Focus::Samples;

        let raw = self.csv_path.clone();
        let Some(resolved) = resolve_csv_path(&raw, &self.project_root) else {
            let hint = self
                .project_root
                .join("samples")
                .join("gt_sensors_demo.csv");
            self.status = format!(
                "CSV not found: {:?}  ·  root={}  ·  try Enter on a sample  ·  hint {}",
                normalize_path_text(&raw),
                self.project_root.display(),
                hint.display()
            );
            self.push_log(self.status.clone());
            return;
        };
        let csv = resolved.display().to_string();
        self.csv_path = csv.clone();
        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        self.busy = true;
        self.status = format!("Running… {csv}");
        self.push_log(format!("START {csv} mode={}", self.mode.label()));
        self.scroll_plots = 0;
        self.scroll_report = 0;

        let python = self.python.clone();
        let root = self.project_root.clone();
        let mode = self.mode.label().to_string();
        let context = self.context.clone();
        let no_dl = env::var("GT_NO_DOWNLOAD").ok().as_deref() == Some("1");

        thread::spawn(move || {
            let app_py = root.join("app.py");
            if !app_py.is_file() {
                let _ = tx.send(Err(anyhow!(
                    "app.py not found at {} — set GT_HARNESS_ROOT to the gt_harness folder",
                    app_py.display()
                )));
                return;
            }
            let mut cmd = Command::new(&python);
            cmd.arg(&app_py)
                .arg("--json-once")
                .arg(&csv)
                .arg("--mode")
                .arg(&mode)
                .arg("--context")
                .arg(&context)
                .current_dir(&root)
                .env("GT_HARNESS_ROOT", &root)
                // Keep child quiet on the shared console (Windows)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            if no_dl {
                cmd.arg("--no-download");
                cmd.env("GT_NO_DOWNLOAD", "1");
            }
            // Avoid second GGUF pass slowing UI demo unless user set it
            if env::var("GT_FULL_REFLECTION").is_err() {
                cmd.env("GT_FULL_REFLECTION", "0");
            }
            #[cfg(windows)]
            {
                cmd.creation_flags(CREATE_NO_WINDOW);
            }

            let out = cmd.output();
            let res = match out {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
                    if !o.status.success() && stdout.trim().is_empty() {
                        Err(anyhow!(
                            "python exit {:?} · {}",
                            o.status.code(),
                            stderr.chars().take(500).collect::<String>()
                        ))
                    } else {
                        parse_bridge_json(&stdout).map_err(|e| {
                            anyhow!(
                                "{e}\n--- stderr ---\n{}",
                                stderr.chars().take(600).collect::<String>()
                            )
                        })
                    }
                }
                Err(e) => Err(anyhow!(
                    "failed to spawn '{python}': {e}  (set GT_HARNESS_PYTHON to full path)"
                )),
            };
            let _ = tx.send(res);
        });
    }

    fn poll_worker(&mut self) {
        let Some(rx) = self.rx.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(result)) => {
                self.busy = false;
                self.rx = None;
                if result.ok {
                    let ch: Vec<String> = result
                        .proof_channels
                        .iter()
                        .map(|c| format!("{}({:.2})", c.name, c.score))
                        .collect();
                    let msg = format!(
                        "Done in {:.0}s · {} · press 1=plots 2=report",
                        result.elapsed_s.unwrap_or(0.0),
                        if ch.is_empty() {
                            "no plot channels".into()
                        } else {
                            ch.join(", ")
                        }
                    );
                    self.status = msg.clone();
                    self.push_log(msg);
                    self.pane = Pane::Plots;
                    self.last = Some(result);
                } else {
                    let err = result
                        .error
                        .clone()
                        .unwrap_or_else(|| "unknown".into());
                    self.status = format!("Engine error: {err}");
                    self.push_log(self.status.clone());
                    self.last = Some(result);
                }
            }
            Ok(Err(e)) => {
                self.busy = false;
                self.rx = None;
                self.status = format!("Run failed: {e}");
                self.push_log(self.status.clone());
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.busy = false;
                self.rx = None;
                self.status = "Worker disconnected".into();
            }
        }
    }

    fn plots_text(&self) -> String {
        self.last
            .as_ref()
            .and_then(|r| r.evidence_ascii.clone())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| {
                "No proof plots yet.\n\n\
                 1. ↑↓ select a sample (or e to edit path)\n\
                 2. Press Enter to load the sample\n\
                 3. Press r or F5 to run diagnosis\n\n\
                 Plots of issue channels appear here with ▲ on flags."
                    .into()
            })
    }

    fn report_text(&self) -> String {
        self.last
            .as_ref()
            .and_then(|r| {
                r.display_markdown
                    .clone()
                    .or_else(|| r.final_report.clone())
            })
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| {
                "No report yet.\n\nRun diagnosis (r / F5) — write-up appears here.".into()
            })
    }
}

fn which_exists(name: &str) -> bool {
    env::var_os("PATH")
        .map(|paths| {
            env::split_paths(&paths).any(|dir| {
                let p = dir.join(name);
                p.is_file()
                    || {
                        #[cfg(windows)]
                        {
                            dir.join(format!("{name}.exe")).is_file()
                        }
                        #[cfg(not(windows))]
                        {
                            false
                        }
                    }
            })
        })
        .unwrap_or(false)
}

fn parse_bridge_json(stdout: &str) -> Result<BridgeResult> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("empty stdout from python --json-once"));
    }
    if let Ok(v) = serde_json::from_str::<BridgeResult>(trimmed) {
        return Ok(v);
    }
    if let Some(start) = trimmed.find('{') {
        let slice = &trimmed[start..];
        if let Ok(v) = serde_json::from_str::<BridgeResult>(slice) {
            return Ok(v);
        }
        if let Some(obj) = extract_json_object(slice) {
            return serde_json::from_str(obj).context("parse extracted JSON");
        }
    }
    Err(anyhow!(
        "could not parse bridge JSON (first 240 chars): {}",
        trimmed.chars().take(240).collect::<String>()
    ))
}

fn extract_json_object(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut start = None;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'{' {
            if depth == 0 {
                start = Some(i);
            }
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                if let Some(st) = start {
                    return std::str::from_utf8(&bytes[st..=i]).ok();
                }
            }
        }
    }
    None
}

// ── Draw ───────────────────────────────────────────────────────────────────

fn ui(f: &mut Frame, app: &mut App) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(12),
            Constraint::Length(4),
            Constraint::Length(1),
        ])
        .split(f.area());

    let header = Paragraph::new(Text::from(vec![
        Line::from(vec![
            Span::styled(
                " GT HARNESS ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(app.chips_line(), Style::default().fg(Color::Gray)),
        ]),
        Line::from(Span::styled(
            format!(" root {} ", app.project_root.display()),
            Style::default().fg(Color::DarkGray),
        )),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" command "),
    );
    f.render_widget(header, root[0]);

    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(38), Constraint::Percentage(62)])
        .split(root[1]);

    draw_setup(f, app, main[0]);
    draw_results(f, app, main[1]);

    let st_style = if app.busy {
        Style::default().fg(Color::Yellow)
    } else if app.status.to_ascii_lowercase().contains("not found")
        || app.status.to_ascii_lowercase().contains("failed")
        || app.status.to_ascii_lowercase().contains("error")
    {
        Style::default().fg(Color::Red)
    } else {
        Style::default().fg(Color::Green)
    };
    let log_tail: String = app
        .log_lines
        .iter()
        .rev()
        .take(2)
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join(" │ ");
    let status = Paragraph::new(Text::from(vec![
        Line::from(Span::styled(app.status.as_str(), st_style)),
        Line::from(Span::styled(log_tail, Style::default().fg(Color::DarkGray))),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" status ")
            .border_style(Style::default().fg(Color::DarkGray)),
    )
    .wrap(Wrap { trim: true });
    f.render_widget(status, root[2]);

    let help = match app.focus {
        Focus::Samples => {
            " ↑↓ sample  Enter load  r/F5 Run  m Mode  e edit path  c context  1 plots 2 report  q Quit "
        }
        Focus::PathEdit => " PATH EDIT · type path  Enter save  Esc cancel ",
        Focus::ContextEdit => " CONTEXT EDIT · type notes  Esc done ",
    };
    f.render_widget(
        Paragraph::new(help).style(Style::default().fg(Color::DarkGray)),
        root[3],
    );
}

fn draw_setup(f: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // path
            Constraint::Length(3), // mode
            Constraint::Min(6),    // samples
            Constraint::Length(4), // context
        ])
        .split(area);

    let path_title = if app.focus == Focus::PathEdit {
        " CSV path · EDITING "
    } else {
        " CSV path "
    };
    let path_border = if app.focus == Focus::PathEdit {
        Color::Green
    } else {
        Color::Gray
    };
    let path_display = if app.csv_path.is_empty() {
        "(no path — pick a sample below)".to_string()
    } else {
        app.csv_path.clone()
    };
    f.render_widget(
        Paragraph::new(path_display)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(path_title)
                    .border_style(Style::default().fg(path_border)),
            )
            .wrap(Wrap { trim: true }),
        chunks[0],
    );

    f.render_widget(
        Paragraph::new(format!(" {}   (m to toggle)", app.mode.label()))
            .style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" mode ")
                    .border_style(Style::default().fg(Color::Yellow)),
            ),
        chunks[1],
    );

    let items: Vec<ListItem> = app
        .samples
        .iter()
        .map(|s| ListItem::new(s.label.as_str()))
        .collect();
    let sample_border = if app.focus == Focus::Samples {
        Color::Cyan
    } else {
        Color::DarkGray
    };
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(
                    " samples ({}) · ↑↓ Enter ",
                    app.samples.len()
                ))
                .border_style(Style::default().fg(sample_border)),
        )
        .highlight_style(
            Style::default()
                .bg(Color::Cyan)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");
    f.render_stateful_widget(list, chunks[2], &mut app.sample_state);

    let ctx_title = if app.focus == Focus::ContextEdit {
        " context · EDITING "
    } else {
        " context (c to edit) "
    };
    let ctx_border = if app.focus == Focus::ContextEdit {
        Color::Green
    } else {
        Color::Gray
    };
    let ctx_body = if app.context.is_empty() && app.focus != Focus::ContextEdit {
        "SOE / alarms / notes…".to_string()
    } else {
        app.context.clone()
    };
    f.render_widget(
        Paragraph::new(ctx_body)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(ctx_title)
                    .border_style(Style::default().fg(ctx_border)),
            )
            .wrap(Wrap { trim: false }),
        chunks[3],
    );
}

fn draw_results(f: &mut Frame, app: &App, area: Rect) {
    let title = match app.pane {
        Pane::Plots => " RESULTS · proof plots (1) ",
        Pane::Report => " RESULTS · write-up (2) ",
    };
    let body = match app.pane {
        Pane::Plots => app.plots_text(),
        Pane::Report => app.report_text(),
    };
    let scroll = match app.pane {
        Pane::Plots => app.scroll_plots,
        Pane::Report => app.scroll_report,
    };
    let border = match app.pane {
        Pane::Plots => Color::Yellow,
        Pane::Report => Color::Magenta,
    };
    f.render_widget(
        Paragraph::new(body)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title)
                    .border_style(Style::default().fg(border)),
            )
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        area,
    );

    if app.busy {
        let popup = centered_rect(50, 30, area);
        f.render_widget(Clear, popup);
        f.render_widget(
            Paragraph::new("Diagnosing…\n\nPython engine (json-once)\nThis can take 1–3 min with GGUF")
                .alignment(Alignment::Center)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" busy ")
                        .border_style(Style::default().fg(Color::Yellow)),
                ),
            popup,
        );
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let root = find_project_root();
    env::set_var("GT_HARNESS_ROOT", &root);

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let res = run_app(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(e) = res {
        eprintln!("ratatui error: {e:#}");
        return Err(e);
    }
    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, app: &mut App) -> Result<()> {
    loop {
        app.poll_worker();
        terminal.draw(|f| ui(f, app))?; // app mut for list state

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        // Windows Terminal often sends both Press and Repeat; ignore Release
        if key.kind == KeyEventKind::Release {
            continue;
        }

        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

        // Global quit
        if matches!(key.code, KeyCode::Esc) && app.focus == Focus::Samples {
            return Ok(());
        }
        if ctrl && matches!(key.code, KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Char('c'))
        {
            return Ok(());
        }

        // Editing modes: only text + Esc/Enter
        if app.focus == Focus::PathEdit {
            match key.code {
                KeyCode::Esc => {
                    app.focus = Focus::Samples;
                    app.status = "Path edit cancelled".into();
                }
                KeyCode::Enter => {
                    if let Some(p) = resolve_csv_path(&app.csv_path, &app.project_root) {
                        app.csv_path = p.display().to_string();
                        app.status = format!("Path set: {}", app.csv_path);
                        app.push_log(app.status.clone());
                    } else {
                        app.status = format!(
                            "Path not found yet (will retry on Run): {}",
                            app.csv_path
                        );
                    }
                    app.focus = Focus::Samples;
                }
                KeyCode::Backspace => {
                    app.csv_path.pop();
                }
                KeyCode::Char(c) if !ctrl => {
                    app.csv_path.push(c);
                }
                _ => {}
            }
            continue;
        }
        if app.focus == Focus::ContextEdit {
            match key.code {
                KeyCode::Esc | KeyCode::Enter => {
                    app.focus = Focus::Samples;
                    app.status = "Context saved".into();
                }
                KeyCode::Backspace => {
                    app.context.pop();
                }
                KeyCode::Char(c) if !ctrl => {
                    app.context.push(c);
                }
                _ => {}
            }
            continue;
        }

        // Navigation mode (Samples) — keys never corrupt the path
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => return Ok(()),
            KeyCode::F(5) | KeyCode::Char('r') | KeyCode::Char('R') => {
                if !app.busy {
                    app.start_diagnosis();
                }
            }
            KeyCode::Char('m') | KeyCode::Char('M') | KeyCode::F(6) => {
                if !app.busy {
                    app.mode = app.mode.toggle();
                    app.status = format!("Mode → {}", app.mode.label());
                }
            }
            KeyCode::Char('e') | KeyCode::Char('E') => {
                if !app.busy {
                    app.focus = Focus::PathEdit;
                    app.status = "Editing CSV path — type, Enter to confirm, Esc cancel".into();
                }
            }
            KeyCode::Char('c') | KeyCode::Char('C') if !ctrl => {
                if !app.busy {
                    app.focus = Focus::ContextEdit;
                    app.status = "Editing context — type notes, Esc when done".into();
                }
            }
            KeyCode::Char('1') | KeyCode::F(1) => app.pane = Pane::Plots,
            KeyCode::Char('2') | KeyCode::F(2) => app.pane = Pane::Report,
            KeyCode::Up | KeyCode::Char('k') => {
                if app.pane == Pane::Plots || app.pane == Pane::Report {
                    // Prefer sample list when focus samples
                }
                let i = app.sample_state.selected().unwrap_or(0);
                let next = i.saturating_sub(1);
                app.sample_state.select(Some(next));
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let i = app.sample_state.selected().unwrap_or(0);
                let max = app.samples.len().saturating_sub(1);
                let next = (i + 1).min(max);
                app.sample_state.select(Some(next));
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                app.apply_selected_sample();
            }
            KeyCode::PageUp => match app.pane {
                Pane::Plots => app.scroll_plots = app.scroll_plots.saturating_sub(5),
                Pane::Report => app.scroll_report = app.scroll_report.saturating_sub(5),
            },
            KeyCode::PageDown => match app.pane {
                Pane::Plots => app.scroll_plots = app.scroll_plots.saturating_add(5),
                Pane::Report => app.scroll_report = app.scroll_report.saturating_add(5),
            },
            // Scroll results with Ctrl+arrows
            KeyCode::Left if ctrl => {
                app.scroll_plots = app.scroll_plots.saturating_sub(3);
                app.scroll_report = app.scroll_report.saturating_sub(3);
            }
            KeyCode::Right if ctrl => {
                app.scroll_plots = app.scroll_plots.saturating_add(3);
                app.scroll_report = app.scroll_report.saturating_add(3);
            }
            _ => {}
        }
    }
}
