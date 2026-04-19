// archive_fetcher.js
//
// Max [js] externals for pulling videos from the SQLite archive built by
// archive_indexer.py. Heat-aware random selection + asynchronous queue.
//
// Messages:
//   open <path>         open SQLite database
//   close               close database
//   heat <float 0..1>   pick a random video at the heat bucket, output "path <str>"
//   bucket <int 0..4>   same but explicit bucket
//   min_duration <sec>  set minimum duration filter
//   query <sql>         run arbitrary SELECT, output rows as list
//   count               output row count
//   prefetch <n>        preload n paths into the local queue
//   next                pop next prefetched path -> "path <str>"
//   stats               output [count, min_heat, max_heat] etc
//
// Outputs:
//   outlet 0: messages ("path <str>", "count <n>", ...)
//   outlet 1: bangs on state transitions (opened, closed, empty)

autowatch = 1;
inlets  = 1;
outlets = 2;

var SQLITE = null;     // Max 9 exposes SQLite via `sqlite` or `sqlite3` JS bindings
var DB     = null;
var QUEUE  = [];
var MIN_DUR = 0.0;

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

function count() {
	if (!DB) return;
	var r = DB.exec("SELECT COUNT(*) AS n FROM videos WHERE duration >= " + MIN_DUR);
	outlet(0, "count", r.length ? r[0].n : 0);
}

function stats() {
	if (!DB) return;
	var r = DB.exec(
		"SELECT COUNT(*) AS n, MIN(organic) AS mn, MAX(organic) AS mx, " +
		"AVG(organic) AS avg FROM videos WHERE duration >= " + MIN_DUR);
	if (r.length) {
		outlet(0, "stats", r[0].n, r[0].mn, r[0].mx, r[0].avg);
	}
}

function bucket(b) {
	if (!DB) return;
	b = Math.max(0, Math.min(4, parseInt(b, 10)));
	// select random row at this heat bucket (sample-with-replacement style)
	var sql =
		"SELECT path FROM videos " +
		"WHERE heat_bucket = " + b + " AND duration >= " + MIN_DUR + " " +
		"ORDER BY RANDOM() LIMIT 1";
	var r = DB.exec(sql);
	if (r.length) {
		outlet(0, "path", r[0].path);
	} else {
		outlet(1, "empty", b);
	}
}

function heat(h) {
	h = parseFloat(h);
	if (isNaN(h)) return;
	bucket(Math.floor(h * 5));
}

function prefetch(n) {
	if (!DB) return;
	n = Math.max(1, parseInt(n, 10) || 8);
	var r = DB.exec(
		"SELECT path FROM videos WHERE duration >= " + MIN_DUR +
		" ORDER BY RANDOM() LIMIT " + n);
	QUEUE = r.map(function (row) { return row.path; });
	outlet(0, "prefetched", QUEUE.length);
}

function next() {
	if (!QUEUE.length) {
		outlet(1, "empty");
		return;
	}
	outlet(0, "path", QUEUE.shift());
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
