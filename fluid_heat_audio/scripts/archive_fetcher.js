// archive_fetcher.js
//
// Max [js] externals for pulling videos from the SQLite archive built by
// archive_indexer.py. Heat-aware random selection + asynchronous queue.
//
// Messages:
//   open <path>               open SQLite database
//   close                     close database
//   role <str>                filter by role ('texture','velocity','both','')
//   channel <str>             filter by channel name, '' = any
//   allow_remote <0|1>        include rows with remote=1 (resolver streams them)
//   only_remote  <0|1>        debug: only return remote rows
//   heat <float 0..1>         pick a random video at the heat bucket
//   bucket <int 0..4>         same but explicit bucket
//   energy <lo> <hi>          pick a random video with lo <= energy <= hi
//   viscosity <lo> <hi>       pick a random video with lo <= viscosity <= hi
//   match <heat> <energy> <visc>  weighted multi-criterion (sql ORDER BY)
//   min_duration <sec>        minimum duration filter (NULL durations pass)
//   query <sql>               run arbitrary SELECT, output rows as list
//   count                     output row count for current filter
//   prefetch <n>              preload n paths into the local queue
//   next                      pop next prefetched path
//   stats                     output [count, min_heat, max_heat] etc
//
// Outputs:
//   outlet 0:
//      "path <local-file>"      -> ready to read with jit.movie
//      "resolve <url>"          -> needs archive_resolver.py (yt-dlp + LRU cache)
//      "count <n>" / "stats ..." / etc
//   outlet 1: bangs on state transitions (opened, closed, empty, error)

autowatch = 1;
inlets  = 1;
outlets = 2;

var SQLITE = null;     // Max 9 exposes SQLite via `sqlite` or `sqlite3` JS bindings
var DB     = null;
var QUEUE  = [];
var MIN_DUR = 0.0;
var ROLE    = "";      // "" means do not filter by role
var CHANNEL = "";      // "" means do not filter by channel
var ALLOW_REMOTE = 1;  // 0 -> only local files; 1 -> remote rows allowed (resolver active)
var ONLY_REMOTE  = 0;  // 1 -> only return remote rows (debug)

function _emit(row) {
	if (!row) return false;
	if (row.remote && parseInt(row.remote, 10) === 1) {
		// remote row - the resolver will turn this into a local path
		outlet(0, "resolve", String(row.path));
	} else {
		outlet(0, "path", String(row.path));
	}
	return true;
}

function _filterSQL(extra) {
	var clauses = ["(duration IS NULL OR duration >= " + MIN_DUR + ")"];
	if (ROLE && ROLE !== "" && ROLE !== "both") {
		clauses.push("(role = '" + ROLE + "' OR role = 'both')");
	}
	if (CHANNEL && CHANNEL !== "") {
		clauses.push("channel = '" + CHANNEL + "'");
	}
	if (!ALLOW_REMOTE) clauses.push("remote = 0");
	if (ONLY_REMOTE)   clauses.push("remote = 1");
	if (extra && extra.length) {
		for (var i = 0; i < extra.length; ++i) clauses.push(extra[i]);
	}
	return "WHERE " + clauses.join(" AND ");
}

function allow_remote(b) { ALLOW_REMOTE = parseInt(b, 10) ? 1 : 0; }
function only_remote(b)  { ONLY_REMOTE  = parseInt(b, 10) ? 1 : 0; }

function loadbang() {
	try {
		SQLITE = new SQLite();
	} catch (e) {
		post("archive_fetcher: Max SQLite JS bindings unavailable -- " + e + "\n");
	}
}

function open(dbpath) {
	if (!SQLITE) { outlet(0, "error", "sqlite binding missing"); return; }
	try {
		DB = SQLITE.open(dbpath, true /* read-only */);
		outlet(1, "opened");
		stats();
	} catch (e) {
		outlet(0, "error", String(e));
	}
}

function close() {
	if (DB) { DB.close(); DB = null; outlet(1, "closed"); }
}

function min_duration(s) { MIN_DUR = Math.max(0, parseFloat(s) || 0); }
function role(r)    { ROLE    = String(r || ""); }
function channel(c) { CHANNEL = String(c || ""); }

function count() {
	if (!DB) return;
	var r = DB.exec("SELECT COUNT(*) AS n FROM videos " + _filterSQL());
	outlet(0, "count", r.length ? r[0].n : 0);
}

function stats() {
	if (!DB) return;
	var r = DB.exec(
		"SELECT COUNT(*) AS n, MIN(organic) AS mn, MAX(organic) AS mx, " +
		"AVG(organic) AS avg FROM videos " + _filterSQL());
	if (r.length) {
		outlet(0, "stats", r[0].n, r[0].mn, r[0].mx, r[0].avg);
	}
}

function bucket(b) {
	if (!DB) return;
	b = Math.max(0, Math.min(4, parseInt(b, 10)));
	var sql =
		"SELECT path, remote FROM videos " +
		_filterSQL(["heat_bucket = " + b]) +
		" ORDER BY RANDOM() LIMIT 1";
	var r = DB.exec(sql);
	if (!_emit(r[0])) outlet(1, "empty", b);
}

function heat(h) {
	h = parseFloat(h);
	if (isNaN(h)) return;
	bucket(Math.floor(h * 5));
}

function energy(lo, hi) {
	if (!DB) return;
	lo = parseFloat(lo); hi = parseFloat(hi);
	if (isNaN(lo)) lo = 0;
	if (isNaN(hi)) hi = 1;
	var sql = "SELECT path, remote FROM videos " +
		_filterSQL(["energy BETWEEN " + lo + " AND " + hi]) +
		" ORDER BY RANDOM() LIMIT 1";
	var r = DB.exec(sql);
	if (!_emit(r[0])) outlet(1, "empty", "energy");
}

function viscosity(lo, hi) {
	if (!DB) return;
	lo = parseFloat(lo); hi = parseFloat(hi);
	if (isNaN(lo)) lo = 0;
	if (isNaN(hi)) hi = 1;
	var sql = "SELECT path, remote FROM videos " +
		_filterSQL(["viscosity BETWEEN " + lo + " AND " + hi]) +
		" ORDER BY RANDOM() LIMIT 1";
	var r = DB.exec(sql);
	if (!_emit(r[0])) outlet(1, "empty", "viscosity");
}

// nearest-match over 3 simultaneous audio descriptors - "living query"
function match(h, e, v) {
	if (!DB) return;
	h = parseFloat(h); e = parseFloat(e); v = parseFloat(v);
	if (isNaN(h)) h = 0.5;
	if (isNaN(e)) e = 0.5;
	if (isNaN(v)) v = 0.5;
	var rank = "ABS(organic - " + h + ") * 1.0 " +
			   "+ ABS(energy - " + e + ") * 0.9 " +
			   "+ ABS(viscosity - " + v + ") * 0.7 " +
			   "+ (ABS(RANDOM() % 100) / 1000.0)";
	var sql = "SELECT path, remote FROM videos " + _filterSQL() +
		" ORDER BY " + rank + " ASC LIMIT 1";
	var r = DB.exec(sql);
	if (!_emit(r[0])) outlet(1, "empty", "match");
}

function prefetch(n) {
	if (!DB) return;
	n = Math.max(1, parseInt(n, 10) || 8);
	var r = DB.exec("SELECT path, remote FROM videos " + _filterSQL() +
		" ORDER BY RANDOM() LIMIT " + n);
	QUEUE = r.map(function (row) {
		return { path: row.path, remote: parseInt(row.remote, 10) === 1 };
	});
	outlet(0, "prefetched", QUEUE.length);
}

function next() {
	if (!QUEUE.length) {
		outlet(1, "empty");
		return;
	}
	var row = QUEUE.shift();
	if (typeof row === "string") {        // legacy queue entries
		outlet(0, "path", row);
	} else if (row.remote) {
		outlet(0, "resolve", row.path);
	} else {
		outlet(0, "path", row.path);
	}
}

// raw query -- trust your own SQL
function query() {
	if (!DB) return;
	var sql = Array.prototype.slice.call(arguments).join(" ");
	try {
		var r = DB.exec(sql);
		outlet(0, "rows", r.length);
		for (var i = 0; i < r.length; ++i) {
			var row = r[i];
			var msg = ["row", i];
			for (var k in row) { msg.push(k, row[k]); }
			outlet(0, msg);
		}
	} catch (e) {
		outlet(0, "error", String(e));
	}
}
