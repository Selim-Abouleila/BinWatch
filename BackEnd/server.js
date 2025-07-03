
// server.js ---------------------------------------------------------
const express  = require('express');
const multer   = require('multer');
const sharp    = require('sharp');
const path     = require('path');
const fs       = require('fs');
const fetch    = require('node-fetch');    // npm i node-fetch@2
const FormData = require('form-data');     // npm i form-data
const { Pool } = require('pg');            // npm i pg


console.log('❯ DATABASE_URL       =', process.env.DATABASE_URL);


const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});


// ── CONFIG ────────────────────────────────────────────────────────
const FLASK_URL    = `http://${process.env.FLASK_HOST || 'localhost'}:${process.env.FLASK_PORT || '5000'}`;
const PORT = process.env.PORT || 3000;
const FRONTEND_DIR = path.join(__dirname, '..', 'FrontEnd');
const UPLOADS_DIR  = path.join(__dirname, 'uploads');


// Postgres pool (will pick up PGHOST, PGUSER, PGPASSWORD, PGDATABASE, PGPORT)


// Ensure uploads folder exists
fs.mkdirSync(UPLOADS_DIR, { recursive: true });

// App setup
const app = express();
app.use(express.static(FRONTEND_DIR));
app.use('/uploads', express.static(UPLOADS_DIR));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Multer config
const storage = multer.diskStorage({
  destination: UPLOADS_DIR,
  filename:  (req, file, cb) => cb(null, Date.now() + '-' + file.originalname)
});
const upload = multer({ storage });

// Route: serve index.html
app.get('/', (req, res) =>
  res.sendFile(path.join(FRONTEND_DIR, 'index.html'))
);

// Route: upload → local metadata → call Flask → insert into Postgres → respond
app.post('/upload', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'No file uploaded' });
  }

  const localPath = `/uploads/${req.file.filename}`;
  const imagePath = path.join(UPLOADS_DIR, req.file.filename);
  const { annotation, location, date } = req.body;

  let pythonLabel, pyFeat;

  try {
    const form = new FormData();
    form.append('image', fs.createReadStream(imagePath));

    const flaskResp = await fetch(`${FLASK_URL}/classify`, {
      method:  'POST',
      body:    form,
      headers: form.getHeaders(),
      timeout: 30000
    });

    if (!flaskResp.ok) {
      throw new Error(`Flask responded ${flaskResp.status}`);
    }

    const json = await flaskResp.json();
    pythonLabel = json.label;
    pyFeat      = json.features;
  } catch (err) {
    console.error('[/upload] classification error:', err);
    return res.status(502).json({ success: false, error: 'Image classification failed' });
  }

  try {
    // 1. Insert image features
    const insertImageSQL = `
      INSERT INTO public.image_features
        (path, file_size_kb, width, height, mean_r, mean_g, mean_b)
      VALUES ($1, $2, $3, $4, $5, $6, $7)
      RETURNING id
    `;
    const imageParams = [
      localPath,
      Math.round(pyFeat.size_kb),
      pyFeat.width,
      pyFeat.height,
      pyFeat.avg_r,
      pyFeat.avg_g,
      pyFeat.avg_b
    ];
    const imageResult = await pool.query(insertImageSQL, imageParams);
    const imageId = imageResult.rows[0]?.id;

    // 2. Insert into history
    const insertHistorySQL = `
      INSERT INTO public.image_history
        (image_id, path, created_at, annotation, location, label)
      VALUES ($1, $2, $3, $4, $5, $6)
    `;
    await pool.query(insertHistorySQL, [
      imageId,
      localPath,
      date ? new Date(date) : new Date(),
      annotation,
      location,
      pythonLabel
    ]);
  } catch (err) {
    console.error("[/upload] DB insert error:", err);
  }

  res.json({
    success: true,
    imageUrl: localPath,
    label: pythonLabel,
    features: pyFeat
  });
});

app.get('/history', async (req, res) => {
  try {
    const result = await pool.query(`
      SELECT h.path, h.created_at, h.annotation, h.location, h.label
      FROM public.image_history h
      ORDER BY h.created_at DESC
      LIMIT 100
    `);
    res.json(result.rows);
  } catch (err) {
    console.error("[/history] DB error:", err);
    res.status(500).json({ success: false, error: 'Erreur de lecture de la base' });
  }
});


app.listen(PORT, () =>
  console.log(`Server running on http://localhost:${PORT}`)
);
// ------------------------------------------------------------------
