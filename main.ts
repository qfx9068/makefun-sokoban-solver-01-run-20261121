// Push-optimal Sokoban solver: rooms off the hallway are independent (a box never leaves its room),
// so solve each room by BFS over pushes (push-optimal), pruning dead squares and frozen 2x2 blocks.
// Budgets are state counts, never time, so runs are byte-identical.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const D = [[0, -1, 'u'], [0, 1, 'd'], [-1, 0, 'l'], [1, 0, 'r']];
const MAXS = 700000;

function parse(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length, w = Math.max(...rows.map((r) => r.length)), n = w * h;
  const wall = new Uint8Array(n).fill(1), goals = [], boxes = [];
  let player = -1;
  rows.forEach((row, y) => { for (let x = 0; x < row.length; x++) {
    const c = row[x], i = y * w + x;
    if (c === '#') continue;
    wall[i] = 0;
    if ('.*+'.includes(c)) goals.push(i);
    if ('$*'.includes(c)) boxes.push(i);
    if ('@+'.includes(c)) player = i;
  }});
  for (let x = 0; x < w; x++) { wall[x] = 1; wall[(h - 1) * w + x] = 1; }
  for (let y = 0; y < h; y++) { wall[y * w] = 1; wall[y * w + w - 1] = 1; }
  return { w, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which one box alone can be pushed onto a goal (reverse pulls from goals).
function liveSq(w, wall, goals) {
  const live = new Uint8Array(wall.length), q = [...goals];
  for (const g of goals) live[g] = 1;
  for (let i = 0; i < q.length; i++) for (const [dx, dy] of D) {
    const d = dy * w + dx, y = q[i] - d;
    if (!wall[y] && !wall[y - d] && !live[y]) { live[y] = 1; q.push(y); }
  }
  return live;
}

// Flood player's region from `from` around occ boxes, marking seen with stamp; return lowest square.
function reach(w, wall, occ, from, seen, stamp, stack) {
  let low = from, n = 0;
  stack[n++] = from; seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (const q of [p - w, p + w, p - 1, p + 1]) if (!wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
  }
  return low;
}

// Box pushed to t completes a 2x2 of walls/boxes holding a box off goal?
function frozen(w, wall, occ, goal, t) {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1, c = a + w, d = c + 1;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goal[a]) || (occ[b] && !goal[b]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
  }
  return false;
}

function walk(w, wall, occ, from, to) {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1), how = new Array(wall.length), q = [from];
  prev[from] = from;
  for (let i = 0; i < q.length; i++) {
    const p = q[i];
    if (p === to) break;
    for (const [dx, dy, ch] of D) {
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

// Push-optimal BFS on one board. Returns the move string or null.
function bfs(w, wall, goals, boxes0, player0) {
  const n = wall.length, goal = new Uint8Array(n), live = liveSq(w, wall, goals);
  for (const g of goals) goal[g] = 1;
  if (boxes0.some((b) => !live[b])) return null;
  const occ = new Uint8Array(n), seen = new Int32Array(n), mark = new Int32Array(n), stack = new Int32Array(n);
  let stamp = 0;
  const done = (bs) => bs.every((b) => goal[b]);
  const keyOf = (bs, low) => { let k = low; for (const b of bs) k = k * n + b; return k; };
  const states = [boxes0], sPl = [player0], parent = [-1], keys = new Set();
  for (const b of boxes0) occ[b] = 1;
  keys.add(keyOf(boxes0, reach(w, wall, occ, player0, seen, ++stamp, stack)));
  for (const b of boxes0) occ[b] = 0;
  let found = done(boxes0) ? 0 : -1;
  for (let head = 0; found < 0 && head < states.length; head++) {
    const bs = states[head];
    for (const b of bs) occ[b] = 1;
    reach(w, wall, occ, sPl[head], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < bs.length && found < 0; bi++) {
      const b = bs[bi];
      for (const [dx, dy] of D) {
        const d = dy * w + dx, t = b + d;
        if (seen[b - d] !== here || wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0; occ[t] = 1;
        const bad = frozen(w, wall, occ, goal, t);
        let key = -1, next = null;
        if (!bad) {
          next = bs.slice(); next[bi] = t; next.sort((a, z) => a - z);
          key = keyOf(next, reach(w, wall, occ, b, mark, ++stamp, stack));
        }
        occ[t] = 0; occ[b] = 1;
        if (bad || keys.has(key)) continue;
        if (states.length >= MAXS) return null;
        keys.add(key); states.push(next); sPl.push(b); parent.push(head);
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
    const d = to - from, dir = D.find(([dx, dy]) => dy * w + dx === d);
    for (const b of boxes) occ[b] = 1;
    const wk = walk(w, wall, occ, player, from - d);
    for (const b of boxes) occ[b] = 0;
    if (wk === null) return null;
    moves += wk + dir[2].toUpperCase();
    player = from; boxes = nb;
  }
  return moves;
}

// Cut the dead squares of the player's hallway row; each remaining component with boxes is a room.
function roomsOf(L) {
  const { w, wall, goals, boxes, player } = L, live = liveSq(w, wall, goals);
  const row = Math.floor(player / w), cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1), rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const sq = [s];
    part[s] = rooms.length;
    for (let i = 0; i < sq.length; i++) for (const [dx, dy] of D) {
      const q = sq[i] + dy * w + dx;
      if (!wall[q] && !cut[q] && part[q] < 0) { part[q] = rooms.length; sq.push(q); }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) { if (part[g] < 0) return null; rooms[part[g]].goals.push(g); }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

function solve(text) {
  const L = parse(text), rooms = roomsOf(L);
  if (!rooms || rooms.length === 0) return bfs(L.w, L.wall, L.goals, L.boxes, L.player) ?? '';
  let player = L.player, moves = '';
  for (const room of rooms) {
    const wall = L.wall.slice();
    for (const o of rooms) if (o !== room) for (const b of o.done ? o.goals : o.boxes) wall[b] = 1;
    const sub = bfs(L.w, wall, room.goals.slice().sort((a, b) => a - b), room.boxes.slice().sort((a, b) => a - b), player);
    if (sub === null) return bfs(L.w, L.wall, L.goals, L.boxes, L.player) ?? '';
    moves += sub;
    for (const c of sub.toLowerCase()) player += c === 'u' ? -L.w : c === 'd' ? L.w : c === 'l' ? -1 : 1;
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${solve(readFileSync(join(I, id, 'level.txt'), 'utf8'))}\n`);
}
