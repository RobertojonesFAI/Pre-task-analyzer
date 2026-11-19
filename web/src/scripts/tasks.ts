// Task explorer: a site's past tasks, what the task model predicted for them (without having seen
// that competition) against what happened, and "what if" changes evaluated in the browser.

// A tree is a leaf value or [feature, threshold, missing goes left (1/0), left, right]
type Tree = number | [number, number, number, Tree, Tree];
type Model = { features: string[]; trees: Tree[] };
type Metrics = Record<string, Record<string, { mae: number; time_mae_min?: number; within_15_min?: number;
  within_30_min?: number; within_10_points?: number }>>;
type TaskModel = { targets: { winner_kmh: Model; goal_share: Model }; metrics: Metrics };
type Task = {
  key: string; date: string; competition: string | null; task: string | null; km: number | null; pilots: number;
  winner_h: number | null; goal: number; pred_kmh: number | null; pred_goal: number | null;
  base_kmh: number | null; base_goal: number | null; x: (number | null)[];
  route?: [number, number, number, string][];   // lat, lon, cylinder radius (m), type
};
type SiteTasks = { features: string[]; tasks: Task[] };

let model: TaskModel | null = null;
let onSelect: (route: Task["route"] | null) => void = () => {};

/** Called with a task's route when one is opened, and with null when the list is shown again. */
export function onTaskSelect(fn: typeof onSelect) {
  onSelect = fn;
}

export async function loadTaskModel(): Promise<TaskModel | null> {
  if (model) return model;
  try {
    const r = await fetch("/data/task_model.json");
    model = r.ok ? ((await r.json()) as TaskModel) : null;
  } catch {
    model = null;
  }
  return model;
}

function evalTree(node: Tree, x: (number | null)[]): number {
  while (Array.isArray(node)) {
    const [f, thr, nanLeft, left, right] = node;
    const v = x[f];
    node = v == null || Number.isNaN(v) ? (nanLeft ? left : right) : v <= thr ? left : right;
  }
  return node;
}

export function predict(m: Model, x: (number | null)[]): number {
  let s = 0;
  for (const t of m.trees) s += evalTree(t, x);
  return s;
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const hm = (h: number | null) => {
  if (h == null || !Number.isFinite(h) || h <= 0) return "–";
  const m = Math.round(h * 60);
  return `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}`;
};
const pct = (v: number | null) => (v == null || !Number.isFinite(v) ? "–" : `${Math.round(v * 100)}%`);
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

// the weather of the day shown on a task card: feature, label, unit, decimals
const WEATHER: [string, string, string, number][] = [
  ["blh_mean", "Boundary layer", "m", 0],
  ["usable_m", "Usable height over the route", "m", 0],
  ["cloud_base_above_m", "Cloud base over the route", "m", 0],
  ["sensible_heat_wm2", "Heat from the ground", "W/m²", 0],
  ["sunshine_wm2", "Sunshine", "W/m²", 0],
  ["cloud_total", "Cloud cover", "%", 0],
  ["cape", "CAPE", "J/kg", 0],
  ["wind_700", "Wind at 700 hPa (~3,000 m)", "m/s", 1],
  ["tailwind_700", "Tailwind along the route", "m/s", 1],
  ["crosswind_700", "Crosswind along the route", "m/s", 1],
  ["lapse_850_700", "Lapse rate 850–700 hPa", "°C/km", 1],
  ["rain_mm", "Rain during the task", "mm", 1],
];

export function modelSummary(m: TaskModel): string {
  const w = m.metrics["winner_kmh|new competition"];
  const g = m.metrics["goal_share|new competition"];
  if (!w || !g) return "";
  const a = w["without weather"], b = w["with weather"];
  return `Task model, on competitions it never saw: winner's time within <b>${Math.round(b.time_mae_min ?? 0)} min</b> on average
    (without weather ${Math.round(a.time_mae_min ?? 0)}), ${pct(b.within_30_min ?? null)} of tasks within 30 min; goal share within
    <b>${Math.round(100 * g["with weather"].mae)} points</b> (without weather ${Math.round(100 * g["without weather"].mae)}).`;
}

export async function renderTasks(siteId: string, box: HTMLElement) {
  box.innerHTML = "";
  let data: SiteTasks | null = null;
  try {
    const r = await fetch(`/data/tasks/${siteId}.json`);
    data = r.ok ? ((await r.json()) as SiteTasks) : null;
  } catch {
    data = null;
  }
  const m = await loadTaskModel();
  if (!data || !data.tasks.length) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const list = () => {
    onSelect(null);
    box.innerHTML = `<h3>Tasks <span class="count">${data!.tasks.length}</span></h3>
      <p class="hint">Predicted by the task model before seeing each competition, against what happened.</p>
      <ol class="tasks">${data!.tasks.map((t, i) => {
        const pred = t.pred_kmh && t.km ? t.km / t.pred_kmh : null;
        return `<li><button data-i="${i}"><span class="n">${esc(t.competition || "")}</span>
          <span class="c">${esc(t.task || "")}</span>
          <span class="d">${t.date} · ${t.km ?? "–"} km · winner ${hm(t.winner_h)} (pred. ${hm(pred)}) · goal ${pct(t.goal)} (pred. ${pct(t.pred_goal)})</span></button></li>`;
      }).join("")}</ol>`;
    box.querySelectorAll<HTMLButtonElement>("button[data-i]").forEach((b) =>
      b.addEventListener("click", () => card(Number(b.dataset.i))));
  };
  const card = (i: number) => {
    const t = data!.tasks[i];
    onSelect(t.route && t.route.length ? t.route : null);
    const f = data!.features;
    const at = (name: string) => f.indexOf(name);
    const ref = (v: number | null) => (v == null ? null : v);
    const kmIdx = at("ss_km"), startIdx = at("start_solar_h");
    const km0 = t.km ?? (kmIdx >= 0 ? (t.x[kmIdx] as number) : null);
    const start0 = startIdx >= 0 ? (t.x[startIdx] as number | null) : null;
    const weather = WEATHER.filter(([k]) => at(k) >= 0 && t.x[at(k)] != null).map(([k, label, unit, d]) => {
      let v = t.x[at(k)] as number;
      if (unit === "%") v *= 100;
      return `<div><dt>${label}</dt><dd>${v.toFixed(d)} <span class="u">${unit}</span></dd></div>`;
    }).join("");
    box.innerHTML = `<button class="back" id="t-back">← Tasks</button>
      <h3 class="task-title">${esc(t.competition || "")}</h3>
      <p class="meta">${esc(t.task || "")} · ${t.date} · ${t.km ?? "–"} km · ${t.pilots} pilots</p>
      <div class="compare">
        <div class="col"><span class="lbl">Winner's time</span>
          <span class="big">${hm(t.winner_h)}</span><span class="sub">real</span>
          <span class="big pred">${hm(t.pred_kmh && t.km ? t.km / t.pred_kmh : null)}</span><span class="sub">predicted with weather</span>
          <span class="small">${hm(t.base_kmh && t.km ? t.km / t.base_kmh : null)} without weather</span></div>
        <div class="col"><span class="lbl">In goal</span>
          <span class="big">${pct(t.goal)}</span><span class="sub">real</span>
          <span class="big pred">${pct(ref(t.pred_goal))}</span><span class="sub">predicted with weather</span>
          <span class="small">${pct(ref(t.base_goal))} without weather</span></div>
      </div>
      ${m ? `<div class="whatif">
        <h4>What if the committee changed the task?</h4>
        <label>Speed-section distance <b id="w-km"></b>
          <input id="r-km" type="range" min="${Math.round((km0 ?? 60) * 0.5)}" max="${Math.round((km0 ?? 60) * 1.5)}" step="1" value="${Math.round(km0 ?? 60)}" ${km0 == null ? "disabled" : ""}></label>
        <label>Start time (solar) <b id="w-start"></b>
          <input id="r-start" type="range" min="${start0 == null ? 11 : Math.max(9, Math.round(start0 - 2))}" max="${start0 == null ? 15 : Math.min(17, Math.round(start0 + 2))}" step="0.25" value="${start0 ?? 13}" ${start0 == null ? "disabled" : ""}></label>
        <div class="whatif-out"><div><span class="lbl">Winner's time</span><b id="w-time"></b></div>
          <div><span class="lbl">In goal</span><b id="w-goal"></b></div></div>
        <p class="hint">Starts from the prediction above and adds the change the model sees, evaluated in your browser. The weather stays as it was on the day.</p>
      </div>` : ""}
      ${weather ? `<h4>Weather along the route (ERA5)</h4><dl class="wx">${weather}</dl>` : ""}`;
    box.querySelector("#t-back")!.addEventListener("click", list);
    if (!m) return;
    const rKm = box.querySelector<HTMLInputElement>("#r-km")!, rStart = box.querySelector<HTMLInputElement>("#r-start")!;
    // start from the honest prediction (made without this competition) and apply the change the
    // model (trained on every task) sees between the task as set and the task as changed
    const kmh0 = predict(m.targets.winner_kmh, t.x), goal0 = predict(m.targets.goal_share, t.x);
    const kmSet = rKm.value, startSet = rStart.value;   // the task as set: no change until a slider moves
    const update = () => {
      const x = t.x.slice();
      const km = Number(rKm.value), st = Number(rStart.value);
      if (kmIdx >= 0 && km0 != null && rKm.value !== kmSet) x[kmIdx] = km;
      if (startIdx >= 0 && start0 != null && rStart.value !== startSet) x[startIdx] = st;
      const kmh = (t.pred_kmh ?? kmh0) * (predict(m.targets.winner_kmh, x) / kmh0);
      const goal = clamp01((t.pred_goal ?? goal0) + predict(m.targets.goal_share, x) - goal0);
      box.querySelector("#w-km")!.textContent = km0 == null ? "–" : `${km} km`;
      box.querySelector("#w-start")!.textContent = start0 == null ? "–" : hm(st).replace("–", "");
      box.querySelector("#w-time")!.textContent = km0 == null ? "–" : hm((rKm.value === kmSet ? km0 : km) / kmh);
      box.querySelector("#w-goal")!.textContent = pct(goal);
    };
    rKm.addEventListener("input", update);
    rStart.addEventListener("input", update);
    update();
    box.scrollIntoView({ block: "start", behavior: "smooth" });
  };
  list();
}
