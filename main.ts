// Push-optimal Sokoban solver. Levels from this generator are a hallway with independent rooms off it:
// a box can never leave its room, so the push optimum of a level is the sum of its rooms' optima.
// Each room (at most 4 boxes) is solved by breadth-first search over pushes, which is push-optimal by
// construction; states are (sorted boxes, player's reachable region) and two deadlock shapes are pruned:
// a box on a square from which no goal is reachable by pushes alone, and a frozen 2x2 block off goal.
// All budgets are counted in states, never in time, so two runs write identical bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DIRS = [[0, -1, 'u'], [0, 1, 'd'], [-1, 0, 'l'], [1, 0, 'r']];
const MAXS = 700000;

function parse(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length, w = Math.max(...rows.map((r) => r.length)), n = w * h;
  const wall = new Uint8Array(n).fill(1), goals = [], boxes = [];
  let player = -1;
  rows.forEach((row, y) => { for (let x = 0; x < row.length; x++) {
    const ch = row[x], i = y * w + x;
    if (ch === '#') continue;
    wall[i] = 0;
    if ('.*+'.includes(ch)) goals.push(i);
    if ('$*'.includes(ch)) boxes.push(i);
    if ('@+'.includes(ch)) player = i;
  }});
  for (let x = 0; x < w; x++) { wall[x] = 1; wall[(h - 1) * w + x] = 1; }
  for (let y = 0; y < h; y++) { wall[y * w] = 1; wall[y * w + w - 1] = 1; }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which one box alone can be pushed onto some goal (reverse pulls from the goals).
function liveSquares(w, wall, goals) {
  const live = new Uint8Array(wall.length), q = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < q.length; qi++) for (const [dx, dy] of DIRS) {
    const d = dy * w + dx, y = q[qi] - d;
    if (!wall[y] && !wall[y - d] && !live[y]) { live[y] = 1; q.push(y); }
  }
  return live;
}

// Flood the player's region around `occ` boxes from `from`, marking `seen` with `stamp`; return its lowest square.
function reach(w, wall, occ, from, seen, stamp, stack) {
  let low = from, n = 0;
  stack[n++] = from; seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    const qs = [p - w, p + w, p - 1, p + 1];
    for (let k = 0; k < 4; k++) { const q = qs[k]; if (!wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; } }
  }
  return low;
}

// True when the box just pushed to t completes a 2x2 of walls/boxes holding a box off its goal.
function frozen(w, wall, occ, goal, t) {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1, c = a + w, d = c + 1;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goal[a]) || (occ[b] && !goal[b]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
  }
  return false;
}

// Shortest walk from->to around walls and boxes, as lower-case letters ('' when already there).
function walk(w, wall, occ, from, to) {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1), how = new Array(wall.length), q = [from];
  prev[from] = from;
  for (let qi = 0; qi < q.length; qi++) {
    const p = q[qi];
    if (p === to) break;
    for (const [dx, dy, ch] of DIRS) {
      const s = p + dy * w + dx;
      if (wall[s] || occ[s] || prev[s] !== -1) continue;
      prev[s] = p; how[s] = ch; q.push(s);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

// Push-optimal BFS for one (sub)level: {boxes0, goals, player} on board w/wall. Returns move string or null.
function solveBFS(w, wall, goals, boxes0, player0) {
  const n = wall.length, goal = new Uint8Array(n), live = liveSquares(w, wall, goals);
  for (const g of goals) goal[g] = 1;
  if (boxes0.some((b) => !live[b])) return null;
  const occ = new Uint8Array(n), seen = new Int32Array(n), mark = new Int32Array(n), stack = new Int32Array(n);
  let stamp = 0;
  const done = (bs) => bs.every((b) => goal[b]);
  const keyOf = (bs, low) => { let k = low; for (const b of bs) k = k * n + b; return k; };
  const states = [boxes0], sPlayer = [player0], parent = [-1], seenKeys = new Set();
  for (const b of boxes0) occ[b] = 1;
  seenKeys.add(keyOf(boxes0, reach(w, wall, occ, player0, seen, ++stamp, stack)));
  for (const b of boxes0) occ[b] = 0;
  let found = done(boxes0) ? 0 : -1;
  for (let head = 0; found < 0 && head < states.length; head++) {
    const bs = states[head];
    for (const b of bs) occ[b] = 1;
    reach(w, wall, occ, sPlayer[head], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < bs.length && found < 0; bi++) {
      const b = bs[bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx, t = b + d;
        if (seen[b - d] !== here || wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0; occ[t] = 1;
        const bad = frozen(w, wall, occ, goal, t);
        let key = -1, next = null;
        if (!bad) {
          next = bs.slice(); next[bi] = t; next.sort((a, b2) => a - b2);
          key = keyOf(next, reach(w, wall, occ, b, mark, ++stamp, stack));
        }
        occ[t] = 0; occ[b] = 1;
        if (bad || seenKeys.has(key)) continue;
        if (states.length >= MAXS) return null;
        seenKeys.add(key); states.push(next); sPlayer.push(b); parent.push(head);
        if (done(next)) found = states.length - 1;
      }
    }
    for (const b of bs) occ[b] = 0;
  }
  if (found < 0) return null;
  const path = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '', player = player0, boxes = boxes0;
  for (const s of path) {
    const nb = states[s];
    const from = boxes.find((b) => !nb.includes(b)), to = nb.find((b) => !boxes.includes(b));
    const d = to - from, dir = DIRS.find(([dx, dy]) => dy * w + dx === d);
    for (const b of boxes) occ[b] = 1;
    const wlk = walk(w, wall, occ, player, from - d);
    for (const b of boxes) occ[b] = 0;
    if (wlk === null) return null;
    moves += wlk + dir[2].toUpperCase();
    player = from; boxes = nb;
  }
  return moves;
}

// Split the level into rooms: cut the dead (non-live) squares of the player's hallway row; each remaining
// component holding boxes is a room. Null when the shape is not a clean set of rooms.
function roomsOf(L) {
  const { w, wall, goals, boxes, player } = L, live = liveSquares(w, wall, goals);
  const row = Math.floor(player / w), cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1), rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) for (const [dx, dy] of DIRS) {
      const q = squares[i] + dy * w + dx;
      if (!wall[q] && !cut[q] && part[q] < 0) { part[q] = rooms.length; squares.push(q); }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) { if (part[g] < 0) return null; rooms[part[g]].goals.push(g); }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

function solveLevel(text) {
  const L = parse(text), rooms = roomsOf(L);
  if (rooms === null || rooms.length === 0) {
    return solveBFS(L.w, L.wall, L.goals, L.boxes, L.player) ?? '';
  }
  let player = L.player, moves = '';
  for (const room of rooms) {
    const wall = L.wall.slice();
    for (const other of rooms) if (other !== room) for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    const sub = solveBFS(L.w, wall, room.goals.slice().sort((a, b) => a - b), room.boxes.slice().sort((a, b) => a - b), player);
    if (sub === null) return solveBFS(L.w, L.wall, L.goals, L.boxes, L.player) ?? '';
    moves += sub;
    for (const ch of sub.toLowerCase()) player += ch === 'u' ? -L.w : ch === 'd' ? L.w : ch === 'l' ? -1 : 1;
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
