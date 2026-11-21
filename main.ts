// Sokoban solver for make.fun sokoban-solver-01-run-20261121.
//
// Strategy: the generated levels are a hallway with independent rooms off it
// (one door each, boxes never leave their room, and a level's push-optimum P*
// is the sum of its rooms' optima). So each room is solved push-optimally on
// its own with A* over pushes, and the room solutions are stitched together
// with shortest player walks through the hallway.
//
// Room A*: state = sorted box cells + canonical player cell (the smallest
// cell of the player's reachable region, so states that differ only by where
// the player stands inside one region collapse). Heuristic = minimum-cost
// assignment of boxes to goals over precomputed grid distances (admissible,
// hence the first goal popped is push-optimal). Pruning: static dead squares,
// corner and wall-run deadlocks, and 2x2 freeze deadlocks. All budgets are
// counts of expansions, never the clock, so two runs write the same bytes.
//
// The search is written on flat typed arrays with numerically packed state
// keys (no per-node allocation in the hot loop), so a room's whole state
// space is searched in seconds. If a room search exhausts its budget, a
// weighted A* is retried; a whole-level push A* runs only when the level
// does not decompose at all. Every produced move string is replayed on the
// level before it is written; a replay that fails writes an empty solution
// for that instance.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Expansion budgets (deterministic counts, not wall-clock time).
const ROOM_OPTIMAL_BUDGET = 400_000;
const ROOM_WEIGHTED_BUDGET = 400_000;
const GLOBAL_BUDGET = 150_000;

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];
const INF = 32767;

interface Level {
  w: number;
  h: number;
  wall: boolean[];
  goals: number[];
  goalSet: Set<number>;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall: boolean[] = new Array(w * h).fill(true);
  const goals: number[] = [];
  const goalSet = new Set<number>();
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#';
      if (ch === '.' || ch === '*' || ch === '+') {
        goals.push(i);
        goalSet.add(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

/** Shortest player walk from `from` to `to` over open cells; '' when unreachable. */
function walkPath(
  w: number,
  from: number,
  to: number,
  open: (i: number) => boolean,
): string {
  if (from === to) return '';
  const prev = new Map<number, { p: number; d: number }>();
  const queue = [from];
  prev.set(from, { p: -1, d: -1 });
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi]!;
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d]!;
      const ny = cy + DY[d]!;
      if (nx < 0 || ny < 0) continue;
      const n = ny * w + nx;
      if (!open(n) || prev.has(n)) continue;
      prev.set(n, { p: c, d });
      if (n === to) {
        let s = '';
        let cur = n;
        while (cur !== from) {
          const step = prev.get(cur)!;
          s = LETTER[step.d]! + s;
          cur = step.p;
        }
        return s;
      }
      queue.push(n);
    }
  }
  return '';
}

interface Push {
  box: number;
  dir: number;
}

interface SearchOpts {
  w: number;
  /** Cells the player may stand on (room cells + door outside cell). */
  area: Set<number>;
  /** Cells a box may occupy (room cells only). */
  boxCells: Set<number>;
  goals: number[];
  dead: Set<number>;
  budget: number;
  weight: number;
}

/**
 * Push A* over one room (weight 1: push-optimal). Returns the list of
 * pushes, or null when the budget is exhausted or the room is unsolvable.
 * Hot loop uses flat typed arrays and packed numeric state keys.
 */
function pushAStar(
  opts: SearchOpts,
  startBoxes: number[],
  startPlayer: number,
): Push[] | null {
  const { w, area, boxCells, goals, dead, budget, weight } = opts;
  const W = w;
  const n = startBoxes.length;
  const maxCell = Math.max(...area, ...boxCells) + 1;
  // Neighbour lists (cell -> up to 4 neighbour ids, -1 when none).
  const nbArea: Int32Array[] = [];
  const nbBox: Int32Array[] = [];
  for (let c = 0; c < maxCell; c++) {
    const cx = c % W;
    const cy = Math.floor(c / W);
    const a = new Int32Array(4).fill(-1);
    const b = new Int32Array(4).fill(-1);
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d]!;
      const ny = cy + DY[d]!;
      if (nx < 0 || ny < 0) continue;
      const id = ny * W + nx;
      if (area.has(id)) a[d] = id;
      if (boxCells.has(id)) b[d] = id;
    }
    nbArea.push(a);
    nbBox.push(b);
  }
  const isArea = new Uint8Array(maxCell);
  const isBoxCell = new Uint8Array(maxCell);
  const isGoal = new Uint8Array(maxCell);
  const isDead = new Uint8Array(maxCell);
  for (const c of area) isArea[c] = 1;
  for (const c of boxCells) isBoxCell[c] = 1;
  for (const g of goals) isGoal[g] = 1;
  for (const d of dead) if (d < maxCell) isDead[d] = 1;
  // All-pairs grid distances over boxCells (boxes move only inside the room).
  const dist = new Int16Array(maxCell * maxCell).fill(INF);
  const q = new Int32Array(maxCell);
  for (const a of boxCells) {
    let qh = 0;
    let qt = 0;
    q[qt++] = a;
    dist[a * maxCell + a] = 0;
    while (qh < qt) {
      const c = q[qh++]!;
      const dc = dist[a * maxCell + c]!;
      const ns = nbBox[c]!;
      for (let d = 0; d < 4; d++) {
        const m = ns[d]!;
        if (m < 0 || dist[a * maxCell + m] !== INF) continue;
        dist[a * maxCell + m] = dc + 1;
        q[qt++] = m;
      }
    }
  }
  const dBetween = (a: number, b: number): number => dist[a * maxCell + b]!;
  // Minimum assignment of boxes to goals (n <= 4: all permutations).
  const perms: number[][] = [];
  const buildPerms = (arr: number[], k: number): void => {
    if (k === 1) {
      perms.push(arr.slice());
      return;
    }
    for (let i = 0; i < k; i++) {
      buildPerms(arr, k - 1);
      const j = k % 2 === 0 ? i : 0;
      const t = arr[j]!;
      arr[j] = arr[k - 1]!;
      arr[k - 1] = t;
    }
  };
  buildPerms(goals.map((_, i) => i), n);
  const hCache = new Map<number, number>();
  const pack = (bs: number[]): number => {
    let p = 0;
    for (const b of bs) p = p * maxCell + b;
    return p;
  };
  const heuristic = (bs: number[]): number => {
    const pk = pack(bs);
    const cached = hCache.get(pk);
    if (cached !== undefined) return cached;
    let best = INF;
    for (const perm of perms) {
      let acc = 0;
      for (let k = 0; k < n; k++) {
        const d = dBetween(bs[k]!, goals[perm[k]!]!);
        if (d === INF) {
          acc = INF;
          break;
        }
        acc += d;
      }
      if (acc < best) best = acc;
    }
    hCache.set(pk, best);
    return best;
  };
  // Scratch flood-fill state (stamp arrays, no allocation per call).
  const stamp = new Int32Array(maxCell);
  let tick = 0;
  const floodMin = (start: number, blocked: (c: number) => boolean): number => {
    tick++;
    const t = tick;
    let min = start;
    let head = 0;
    let tail = 0;
    q[tail++] = start;
    stamp[start] = t;
    while (head < tail) {
      const c = q[head++]!;
      if (c < min) min = c;
      const ns = nbArea[c]!;
      for (let d = 0; d < 4; d++) {
        const m = ns[d]!;
        if (m < 0 || stamp[m] === t || blocked(m)) continue;
        stamp[m] = t;
        q[tail++] = m;
      }
    }
    return min;
  };
  const floodHas = (start: number, target: number, blocked: (c: number) => boolean): boolean => {
    tick++;
    const t = tick;
    let head = 0;
    let tail = 0;
    q[tail++] = start;
    stamp[start] = t;
    while (head < tail) {
      const c = q[head++]!;
      if (c === target) return true;
      const ns = nbArea[c]!;
      for (let d = 0; d < 4; d++) {
        const m = ns[d]!;
        if (m < 0 || stamp[m] === t || blocked(m)) continue;
        stamp[m] = t;
        q[tail++] = m;
      }
    }
    return false;
  };
  const inBoxes = (bs: number[], c: number): boolean => {
    for (const b of bs) if (b === c) return true;
    return false;
  };
  const deadlocked = (bs: number[]): boolean => {
    for (const b of bs) {
      if (!isGoal[b]! && isDead[b]!) return true;
    }
    // 2x2 freeze: a 2x2 block of walls/boxes holding a box off a goal.
    for (const b of bs) {
      if (isGoal[b]!) continue;
      const bx = b % W;
      const by = Math.floor(b / W);
      for (let k = 0; k < 4; k++) {
        const ox = k === 1 || k === 3 ? -1 : 0;
        const oy = k >= 2 ? -1 : 0;
        const x0 = bx + ox;
        const y0 = by + oy;
        if (x0 < 0 || y0 < 0) continue;
        let blocked = 0;
        let hasBox = false;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const c = (y0 + dy) * W + (x0 + dx);
            if (c >= maxCell || !isArea[c] || inBoxes(bs, c)) {
              blocked++;
              if (c < maxCell && inBoxes(bs, c)) hasBox = true;
            }
          }
        }
        if (blocked === 4 && hasBox) return true;
      }
    }
    return false;
  };
  const canonPlayer = (player: number, bs: number[]): number =>
    floodMin(player, (c) => inBoxes(bs, c));

  const startCanon = canonPlayer(startPlayer, startBoxes);
  const nodeBoxes: number[] = [pack(startBoxes)];
  const nodePlayer: number[] = [startCanon];
  const nodeG: number[] = [0];
  const nodeParent: number[] = [-1];
  const nodePushBox: number[] = [-1];
  const nodePushDir: number[] = [-1];
  const best = new Map<number, number>();
  best.set(pack(startBoxes) * maxCell + startCanon, 0);
  const h0 = heuristic(startBoxes);
  if (h0 === INF || deadlocked(startBoxes)) return null;
  // Binary heap of [f, h, nodeId]; smaller f, then smaller h, then older node.
  const heap: [number, number, number][] = [[weight * h0, h0, 0]];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  const unpack = (pk: number, out: number[]): void => {
    for (let k = n - 1; k >= 0; k--) {
      out[k] = pk % maxCell;
      pk = Math.floor(pk / maxCell);
    }
  };
  const bs: number[] = new Array(n);
  const nb: number[] = new Array(n);

  let expanded = 0;
  while (heap.length > 0 && expanded < budget) {
    const [, , s] = heapPop();
    unpack(nodeBoxes[s]!, bs);
    const g = nodeG[s]!;
    const key = nodeBoxes[s]! * maxCell + nodePlayer[s]!;
    if (best.get(key) !== s) continue;
    if (bs.every((b) => isGoal[b]!)) {
      const pushes: Push[] = [];
      for (let t = s; t > 0; t = nodeParent[t]!) {
        pushes.push({ box: nodePushBox[t]!, dir: nodePushDir[t]! });
      }
      pushes.reverse();
      return pushes;
    }
    expanded++;
    const pCanon = nodePlayer[s]!;
    for (let bi = 0; bi < n; bi++) {
      const b = bs[bi]!;
      const bx = b % W;
      const by = Math.floor(b / W);
      for (let d = 0; d < 4; d++) {
        const tx = bx + DX[d]!;
        const ty = by + DY[d]!;
        if (tx < 0 || ty < 0) continue;
        const target = ty * W + tx;
        if (target >= maxCell || !isBoxCell[target] || inBoxes(bs, target)) continue;
        const px = bx - DX[d]!;
        const py = by - DY[d]!;
        if (px < 0 || py < 0) continue;
        const from = py * W + px;
        if (from >= maxCell || !isArea[from] || inBoxes(bs, from)) continue;
        if (!floodHas(pCanon, from, (c) => inBoxes(bs, c))) continue;
        for (let k = 0; k < n; k++) nb[k] = bs[k]!;
        nb[bi] = target;
        nb.sort((a, b2) => a - b2);
        if (deadlocked(nb)) continue;
        const h = heuristic(nb);
        if (h === INF) continue;
        const npc = canonPlayer(b, nb);
        const npk = pack(nb);
        const nk = npk * maxCell + npc;
        const seen = best.get(nk);
        if (seen !== undefined && nodeG[seen]! <= g + 1) continue;
        const t = nodeBoxes.length;
        nodeBoxes.push(npk);
        nodePlayer.push(npc);
        nodeG.push(g + 1);
        nodeParent.push(s);
        nodePushBox.push(b);
        nodePushDir.push(d);
        best.set(nk, t);
        heapPush([g + 1 + weight * h, h, t]);
      }
    }
  }
  return null;
}

/** Dead squares of a room: cells no box can ever leave towards a goal. */
function deadSquares(
  w: number,
  h: number,
  roomCells: Set<number>,
  goals: number[],
  wall: boolean[],
): Set<number> {
  const dead = new Set<number>();
  const goalSet = new Set(goals);
  // 1. Cells from which no goal is reachable over room cells.
  const reachGoal = new Set<number>();
  const q: number[] = [];
  for (const g of goals) {
    reachGoal.add(g);
    q.push(g);
  }
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d]!;
      const ny = cy + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const nb = ny * w + nx;
      if (!roomCells.has(nb) || reachGoal.has(nb)) continue;
      reachGoal.add(nb);
      q.push(nb);
    }
  }
  for (const c of roomCells) if (!reachGoal.has(c)) dead.add(c);
  const isWall = (x: number, y: number): boolean => {
    if (x < 0 || y < 0 || x >= w || y >= h) return true;
    return wall[y * w + x]!;
  };
  // 2. Corners (two perpendicular walls) that are not goals.
  for (const c of roomCells) {
    if (goalSet.has(c)) continue;
    const x = c % w;
    const y = Math.floor(c / w);
    const up = isWall(x, y - 1);
    const down = isWall(x, y + 1);
    const left = isWall(x - 1, y);
    const right = isWall(x + 1, y);
    if ((up && left) || (up && right) || (down && left) || (down && right)) dead.add(c);
  }
  // 3. Wall runs: a straight run of cells against one wall, closed at both
  // ends by a wall or a corner and holding no goal, can never be left.
  for (const [dx, dy, wx, wy] of [
    [1, 0, 0, -1],
    [1, 0, 0, 1],
    [0, 1, -1, 0],
    [0, 1, 1, 0],
  ] as const) {
    const seen = new Set<number>();
    for (const c of roomCells) {
      if (seen.has(c) || goalSet.has(c)) continue;
      const x = c % w;
      const y = Math.floor(c / w);
      if (!isWall(x + wx, y + wy)) continue;
      const px = x - dx;
      const py = y - dy;
      if (roomCells.has(py * w + px) && isWall(px + wx, py + wy)) continue;
      const run: number[] = [];
      let cx = x;
      let cy = y;
      let hasGoal = false;
      while (roomCells.has(cy * w + cx) && isWall(cx + wx, cy + wy)) {
        const cc = cy * w + cx;
        run.push(cc);
        seen.add(cc);
        if (goalSet.has(cc)) hasGoal = true;
        cx += dx;
        cy += dy;
      }
      // The run is closed at an end only when the cell just past that end,
      // along the run, is a wall: a neighbouring floor cell lets a box slide
      // off the run (through a gap in the wall, or on to a goal), so a
      // "corner" next to the run does not close it.
      const closedStart = isWall(px, py);
      // ...and at its end when the cell after it is a wall.
      const closedEnd = isWall(cx, cy);
      if (!hasGoal && closedStart && closedEnd) {
        for (const cc of run) dead.add(cc);
      }
    }
  }
  return dead;
}

interface Room {
  cells: Set<number>;
  door: number;
  outside: number;
  boxes: number[];
  goals: number[];
}

/** Split a level into hallway + rooms. Returns null when it does not fit. */
function decompose(lv: Level): Room[] | null {
  const { w, h, wall } = lv;
  const hallRow = Math.floor(lv.player / w);
  const inHall = (i: number): boolean => Math.floor(i / w) === hallRow;
  const open = (i: number): boolean => !wall[i]!;
  const seen = new Set<number>();
  const rooms: Room[] = [];
  const boxRoom = new Map<number, number>();
  for (let i = 0; i < w * h; i++) {
    if (!open(i) || inHall(i) || seen.has(i)) continue;
    // One connected component of non-hallway floor.
    const comp = new Set<number>();
    const stack = [i];
    seen.add(i);
    while (stack.length > 0) {
      const c = stack.pop()!;
      comp.add(c);
      const cx = c % w;
      const cy = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d]!;
        const ny = cy + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const nb = ny * w + nx;
        if (!open(nb) || inHall(nb) || seen.has(nb)) continue;
        seen.add(nb);
        stack.push(nb);
      }
    }
    const boxes = lv.boxes.filter((b) => comp.has(b));
    if (boxes.length === 0) continue;
    const goals = lv.goals.filter((g) => comp.has(g));
    if (goals.length !== boxes.length) return null;
    // Door: the component cell adjacent to a hallway floor cell.
    let door = -1;
    let outside = -1;
    for (const c of [...comp].sort((a, b) => a - b)) {
      const cx = c % w;
      const cy = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d]!;
        const ny = cy + DY[d]!;
        if (ny !== hallRow || nx < 0 || nx >= w) continue;
        const nb = ny * w + nx;
        if (open(nb)) {
          door = c;
          outside = nb;
          break;
        }
      }
      if (door >= 0) break;
    }
    if (door < 0) return null;
    const room: Room = { cells: comp, door, outside, boxes, goals };
    rooms.push(room);
    for (const b of boxes) boxRoom.set(b, rooms.length - 1);
  }
  // Every box must belong to exactly one room.
  if (boxRoom.size !== lv.boxes.length) return null;
  rooms.sort((a, b) => a.outside - b.outside);
  return rooms;
}

/** Turn a room's push list into player moves, walking from `player`. */
function pushesToMoves(
  w: number,
  area: Set<number>,
  startBoxes: number[],
  startPlayer: number,
  pushes: Push[],
): { moves: string; player: number; boxes: number[] } | null {
  let moves = '';
  let player = startPlayer;
  const boxes = startBoxes.slice();
  for (const push of pushes) {
    const from = push.box - (DY[push.dir]! * w + DX[push.dir]!);
    const boxSet = new Set(boxes);
    const path = walkPath(w, player, from, (i) => area.has(i) && !boxSet.has(i));
    if (path === '' && player !== from) return null;
    moves += path;
    moves += LETTER[push.dir]!.toUpperCase();
    const bi = boxes.indexOf(push.box);
    if (bi < 0) return null;
    boxes[bi] = push.box + (DY[push.dir]! * w + DX[push.dir]!);
    boxes.sort((a, b) => a - b);
    player = push.box;
  }
  return { moves, player, boxes };
}

function solveRoom(
  lv: Level,
  room: Room,
  player: number,
  curBoxes: number[],
): { moves: string; player: number; boxes: number[] } | null {
  const area = new Set<number>(room.cells);
  area.add(room.outside);
  const dead = deadSquares(lv.w, lv.h, room.cells, room.goals, lv.wall);
  const opts: SearchOpts = {
    w: lv.w,
    area,
    boxCells: room.cells,
    goals: room.goals,
    dead,
    budget: ROOM_OPTIMAL_BUDGET,
    weight: 1,
  };
  // Walk the player to the room's door first (outside cell). The walk must
  // avoid the boxes where they are NOW (solved rooms already moved theirs
  // onto goals), not where they started.
  const allBoxes = new Set(curBoxes);
  const toDoor = walkPath(
    lv.w,
    player,
    room.outside,
    (i) => !lv.wall[i]! && !allBoxes.has(i),
  );
  if (toDoor === '' && player !== room.outside) return null;
  let cur = player;
  for (const ch of toDoor) {
    const d = LETTER.indexOf(ch);
    cur = cur + DY[d]! * lv.w + DX[d]!;
  }
  let pushes = pushAStar(opts, room.boxes, cur);
  if (!pushes) {
    pushes = pushAStar({ ...opts, budget: ROOM_WEIGHTED_BUDGET, weight: 3 }, room.boxes, cur);
  }
  if (!pushes) return null;
  const conv = pushesToMoves(lv.w, area, room.boxes, cur, pushes);
  if (!conv) return null;
  // Merge the room's new box positions into the level-wide box list.
  const roomBoxSet = new Set(room.boxes);
  const boxes = curBoxes.filter((b) => !roomBoxSet.has(b)).concat(conv.boxes);
  boxes.sort((a, b) => a - b);
  return { moves: toDoor + conv.moves, player: conv.player, boxes };
}

/** Whole-level push A* fallback, used only when a level does not decompose. */
function solveGlobal(lv: Level): string {
  const all = new Set<number>();
  for (let i = 0; i < lv.w * lv.h; i++) if (!lv.wall[i]!) all.add(i);
  const dead = deadSquares(lv.w, lv.h, all, lv.goals, lv.wall);
  const opts: SearchOpts = {
    w: lv.w,
    area: all,
    boxCells: all,
    goals: lv.goals,
    dead,
    budget: GLOBAL_BUDGET,
    weight: 1,
  };
  let pushes = pushAStar(opts, lv.boxes, lv.player);
  if (!pushes) {
    pushes = pushAStar({ ...opts, weight: 3 }, lv.boxes, lv.player);
  }
  if (!pushes) return '';
  const conv = pushesToMoves(lv.w, all, lv.boxes, lv.player, pushes);
  return conv ? conv.moves : '';
}

/** Replay moves on the level; true when every box ends on a goal. */
function replay(lv: Level, moves: string): boolean {
  const boxes = new Set(lv.boxes);
  let p = lv.player;
  for (const ch of moves) {
    const d = LETTER.indexOf(ch.toLowerCase());
    if (d < 0) return false;
    const q = p + DY[d]! * lv.w + DX[d]!;
    if (lv.wall[q]) return false;
    if (boxes.has(q)) {
      const t = q + DY[d]! * lv.w + DX[d]!;
      if (lv.wall[t] || boxes.has(t)) return false;
      boxes.delete(q);
      boxes.add(t);
    }
    p = q;
  }
  for (const b of boxes) if (!lv.goalSet.has(b)) return false;
  return true;
}

function solve(text: string): string {
  const lv = parseLevel(text);
  const rooms = decompose(lv);
  if (rooms) {
    let moves = '';
    let player = lv.player;
    let boxes = lv.boxes.slice();
    let ok = true;
    for (const room of rooms) {
      const r = solveRoom(lv, room, player, boxes);
      if (!r) {
        ok = false;
        break;
      }
      moves += r.moves;
      player = r.player;
      boxes = r.boxes;
    }
    if (ok && replay(lv, moves)) return moves;
    // A room failed: the level cannot be solved this way; an unsolved
    // replay scores 0 either way, so answer empty rather than burn the
    // whole budget on a global search that cannot beat it.
    return '';
  }
  const g = solveGlobal(lv);
  if (g !== '' && replay(lv, g)) return g;
  return '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
