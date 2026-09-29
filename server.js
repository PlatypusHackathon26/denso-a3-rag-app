"use strict";

const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const zlib = require("node:zlib");
const { DatabaseSync } = require("node:sqlite");

const ROOT = __dirname;
const DB_PATH = path.join(ROOT, "factory-chat.db");
const PORT = Number(process.env.PORT || 3000);
const MAX_UPLOAD = 15 * 1024 * 1024;
const RAG_URL = process.env.RAG_URL || "http://127.0.0.1:8000/api/chat";
const db = new DatabaseSync(DB_PATH);
db.exec(`PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE, display_name TEXT NOT NULL,
  password_hash TEXT NOT NULL, salt TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('employee','manager','director')),
  job_title TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY, title TEXT NOT NULL, original_name TEXT NOT NULL,
  content TEXT NOT NULL, min_role TEXT NOT NULL CHECK(min_role IN ('employee','manager','director')),
  uploaded_by INTEGER NOT NULL REFERENCES users(id), created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires_at INTEGER NOT NULL
);`);

const roles = { employee: 0, manager: 1, director: 2 };
const hashPassword = (password, salt) =>
  crypto.scryptSync(password, salt, 64).toString("hex");
const seed = db.prepare("SELECT COUNT(*) AS count FROM users").get();
if (!seed.count) {
  const insert = db.prepare(
    "INSERT INTO users (username,display_name,password_hash,salt,role,job_title) VALUES (?,?,?,?,?,?)",
  );
  for (const user of [
    [
      "nhanvien",
      "Nguyễn Minh An",
      "NhanVien@123",
      "employee",
      "Nhân viên sản xuất",
    ],
    ["quanly", "Trần Thu Hà", "QuanLy@123", "manager", "Quản lý vận hành"],
    ["giamdoc", "Lê Hoàng Nam", "GiamDoc@123", "director", "Giám đốc nhà máy"],
  ]) {
    const salt = crypto.randomBytes(16).toString("hex");
    insert.run(
      user[0],
      user[1],
      hashPassword(user[2], salt),
      salt,
      user[3],
      user[4],
    );
  }
}

function json(res, status, payload, headers = {}) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    ...headers,
  });
  res.end(JSON.stringify(payload));
}
function readBody(req, limit = 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > limit) {
        reject(
          Object.assign(new Error("Dữ liệu vượt giới hạn."), { status: 413 }),
        );
        req.destroy();
      } else chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}
const cookieValue = (req, name) =>
  (req.headers.cookie || "")
    .split(";")
    .map((v) => v.trim())
    .find((v) => v.startsWith(`${name}=`))
    ?.slice(name.length + 1);
function currentUser(req) {
  const raw = cookieValue(req, "factory_session");
  if (!raw) return null;
  const tokenHash = crypto.createHash("sha256").update(raw).digest("hex");
  const row = db
    .prepare(
      "SELECT u.id,u.username,u.display_name,u.role,u.job_title FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires_at > ?",
    )
    .get(tokenHash, Date.now());
  return row || null;
}
function cleanText(text) {
  return text
    .normalize("NFKC")
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, " ")
    .replace(/[^\S\r\n]+/g, " ")
    .replace(/[ \t]*\r?\n[ \t]*/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}
function pdfText(buffer) {
  const source = buffer.toString("latin1");
  const streams = [];
  const re = /<<(.*?)>>\s*stream\r?\n([\s\S]*?)\r?\nendstream/g;
  let match;
  while ((match = re.exec(source))) {
    let data = Buffer.from(match[2], "latin1");
    if (/\/FlateDecode/.test(match[1])) {
      try {
        data = zlib.inflateSync(data);
      } catch {
        try {
          data = zlib.inflateRawSync(data);
        } catch {
          continue;
        }
      }
    }
    streams.push(data.toString("latin1"));
  }
  if (!streams.length) streams.push(source);
  let output = "";
  for (const stream of streams) {
    // Extract PDF text-showing operands from page content streams.
    const tokens =
      /\(((?:\\.|[^\\)])*)\)\s*Tj|\[((?:.|\n)*?)\]\s*TJ|<([\da-fA-F\s]+)>\s*Tj/g;
    let token;
    while ((token = tokens.exec(stream))) {
      const pieces =
        token[2] !== undefined
          ? [
              ...token[2].matchAll(/\(((?:\\.|[^\\)])*)\)|<([\da-fA-F\s]+)>/g),
            ].map((x) =>
              x[1] !== undefined ? unescapePdf(x[1]) : decodeHex(x[2]),
            )
          : [
              token[1] !== undefined
                ? unescapePdf(token[1])
                : decodeHex(token[3]),
            ];
      output += `${pieces.join("")} `;
    }
  }
  return cleanText(output);
}
function unescapePdf(s) {
  return s
    .replace(
      /\\([nrtbf()\\])/g,
      (_, c) => ({ n: "\n", r: "\r", t: "\t", b: "\b", f: "\f" })[c] || c,
    )
    .replace(/\\([0-7]{1,3})/g, (_, o) => String.fromCharCode(parseInt(o, 8)));
}
function decodeHex(s) {
  const hex = s.replace(/\s/g, "");
  if (!hex) return "";
  const bytes = Buffer.from(hex.length % 2 ? `${hex}0` : hex, "hex");
  if (bytes.length > 1 && bytes[0] === 0xfe && bytes[1] === 0xff) {
    let out = "";
    for (let i = 2; i + 1 < bytes.length; i += 2)
      out += String.fromCharCode(bytes.readUInt16BE(i));
    return out;
  }
  return bytes.toString("latin1");
}
function extractUpload(raw) {
  const boundary = raw
    .toString("latin1", 0, Math.min(raw.length, 3000))
    .match(/boundary=(?:"([^"]+)"|([^;\s]+))/)
    ?.slice(1)
    .find(Boolean);
  if (!boundary)
    throw Object.assign(new Error("Thiếu boundary của tệp tải lên."), {
      status: 400,
    });
  const marker = Buffer.from(`--${boundary}`);
  let start = raw.indexOf(marker) + marker.length;
  start = raw.indexOf(Buffer.from("\r\n\r\n"), start);
  if (start < 0)
    throw Object.assign(new Error("Dữ liệu tải lên không hợp lệ."), {
      status: 400,
    });
  const headerStart = raw.indexOf(marker) + marker.length + 2;
  const headers = raw.toString("latin1", headerStart, start);
  const filename = headers.match(/filename="([^"]+)"/i)?.[1];
  const contentStart = start + 4;
  let end = raw.indexOf(Buffer.from(`\r\n--${boundary}`), contentStart);
  if (!filename || end < 0)
    throw Object.assign(new Error("Vui lòng chọn một tệp PDF."), {
      status: 400,
    });
  const name = path.basename(filename).replace(/[\r\n]/g, "");
  const data = raw.subarray(contentStart, end);
  if (
    !name.toLowerCase().endsWith(".pdf") ||
    data.subarray(0, 5).toString() !== "%PDF-"
  )
    throw Object.assign(new Error("Chỉ chấp nhận tệp PDF hợp lệ."), {
      status: 400,
    });
  return { name, data };
}
function visibleDocs(user) {
  return db
    .prepare(
      "SELECT d.id,d.title,d.original_name,d.min_role,d.created_at,u.display_name AS uploader FROM documents d JOIN users u ON u.id=d.uploaded_by WHERE ? >= CASE d.min_role WHEN 'employee' THEN 0 WHEN 'manager' THEN 1 ELSE 2 END ORDER BY d.created_at DESC",
    )
    .all(roles[user.role]);
}
function staticFile(req, res) {
  const pathname = new URL(req.url, "http://localhost").pathname;
  const file =
    pathname === "/" || pathname === "/front_end.html"
      ? "front_end.html"
      : null;
  if (!file) return json(res, 404, { error: "Không tìm thấy trang." });
  res.writeHead(200, {
    "Content-Type": "text/html; charset=utf-8",
    "Cache-Control": "no-store",
  });
  fs.createReadStream(path.join(ROOT, file)).pipe(res);
}
const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    if (
      req.method === "GET" &&
      (url.pathname === "/" || url.pathname === "/front_end.html")
    )
      return staticFile(req, res);
    if (req.method === "POST" && url.pathname === "/api/login") {
      const body = JSON.parse((await readBody(req)).toString("utf8"));
      const u = db
        .prepare("SELECT * FROM users WHERE username=?")
        .get(String(body.username || "").trim());
      if (
        !u ||
        !crypto.timingSafeEqual(
          Buffer.from(hashPassword(String(body.password || ""), u.salt), "hex"),
          Buffer.from(u.password_hash, "hex"),
        )
      )
        return json(res, 401, {
          error: "Tên đăng nhập hoặc mật khẩu không đúng.",
        });
      const token = crypto.randomBytes(32).toString("hex"),
        expiry = Date.now() + 8 * 60 * 60 * 1000;
      db.prepare(
        "INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)",
      ).run(
        crypto.createHash("sha256").update(token).digest("hex"),
        u.id,
        expiry,
      );
      return json(
        res,
        200,
        {
          user: {
            id: u.id,
            username: u.username,
            display_name: u.display_name,
            role: u.role,
            job_title: u.job_title,
          },
        },
        {
          "Set-Cookie": `factory_session=${token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=28800`,
        },
      );
    }
    if (req.method === "POST" && url.pathname === "/api/logout") {
      const raw = cookieValue(req, "factory_session");
      if (raw)
        db.prepare("DELETE FROM sessions WHERE token_hash=?").run(
          crypto.createHash("sha256").update(raw).digest("hex"),
        );
      return json(
        res,
        200,
        { ok: true },
        {
          "Set-Cookie":
            "factory_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0",
        },
      );
    }
    if (req.method === "GET" && url.pathname === "/api/me") {
      const user = currentUser(req);
      return user
        ? json(res, 200, { user })
        : json(res, 401, { error: "Bạn chưa đăng nhập." });
    }
    const user = currentUser(req);
    if (!user)
      return json(res, 401, { error: "Vui lòng đăng nhập để tiếp tục." });
    if (req.method === "GET" && url.pathname === "/api/documents")
      return json(res, 200, { documents: visibleDocs(user) });
    if (req.method === "POST" && url.pathname === "/api/upload") {
      if (roles[user.role] < roles.manager)
        return json(res, 403, {
          error: "Chỉ quản lý và giám đốc được tải tài liệu lên.",
        });
      const { name, data } = extractUpload(await readBody(req, MAX_UPLOAD));
      const content = pdfText(data);
      if (!content)
        return json(res, 422, {
          error:
            "Không trích xuất được văn bản. PDF cần có lớp văn bản (không phải bản scan).",
        });
      const title = path.basename(name, path.extname(name)).slice(0, 180);
      const minRole = ["employee", "manager", "director"].includes(
        url.searchParams.get("access"),
      )
        ? url.searchParams.get("access")
        : "employee";
      const result = db
        .prepare(
          "INSERT INTO documents(title,original_name,content,min_role,uploaded_by) VALUES(?,?,?,?,?)",
        )
        .run(title, name, content, minRole, user.id);
      return json(res, 201, {
        id: Number(result.lastInsertRowid),
        title,
        min_role: minRole,
        characters: content.length,
      });
    }
    if (req.method === "POST" && url.pathname === "/api/chat") {
      const body = JSON.parse(
        (await readBody(req, 32 * 1024 * 1024)).toString("utf8"),
      );
      const question = cleanText(String(body.message || body.question || "")).slice(0, 1000);
      if (!question) return json(res, 400, { error: "Hãy nhập câu hỏi." });
      const docs = db
        .prepare(
          "SELECT title,content,min_role FROM documents WHERE ? >= CASE min_role WHEN 'employee' THEN 0 WHEN 'manager' THEN 1 ELSE 2 END",
        )
        .all(roles[user.role]);
      let ragResponse;
      try {
        ragResponse = await fetch(RAG_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(process.env.RAG_API_TOKEN
              ? { "X-RAG-Token": process.env.RAG_API_TOKEN }
              : {}),
          },
          body: JSON.stringify({
            question,
            mode: ["naive", "local", "global", "hybrid"].includes(body.mode)
              ? body.mode
              : "hybrid",
            role: user.role,
            documents: docs.map(({ title, content }) => ({ title, content })),
          }),
        });
      } catch {
        return json(res, 503, {
          error: "Dịch vụ LightRAG chưa chạy. Hãy khởi động app.py rồi thử lại.",
        });
      }
      const result = await ragResponse.json().catch(() => ({}));
      if (!ragResponse.ok)
        return json(res, ragResponse.status === 401 ? 502 : ragResponse.status, {
          error: result.detail || "LightRAG không xử lý được câu hỏi.",
        });
      return json(res, 200, result);
    }
    return json(res, 404, { error: "Không tìm thấy API." });
  } catch (error) {
    if (!res.headersSent && !res.destroyed)
      json(res, error.status || 500, {
        error: error.status
          ? error.message
          : "Có lỗi máy chủ. Vui lòng thử lại.",
      });
  }
});
server.listen(PORT, "127.0.0.1", () =>
  console.log(`Factory Chat đang chạy tại http://localhost:${PORT}`),
);
