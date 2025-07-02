
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
  // 1. Ensure we have a file
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'No file uploaded' });
  }

  // 2. Build the local URL & filepath
  const localPath = `/uploads/${req.file.filename}`;
  const imagePath = path.join(UPLOADS_DIR, req.file.filename);

  let pythonLabel, pyFeat;
  // 3. Classification + feature extraction
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
    return res.status(502).json({
      success: false,
      error: 'Image classification failed'
    });
  }

  // 4. (Optional) Insert into Postgres; errors here don't block the response
  try {
    const insertSQL = `
      INSERT INTO public.image_features
        (path, file_size_kb, width, height, mean_r, mean_g, mean_b)
      VALUES ($1, $2, $3, $4, $5, $6, $7)
    `;
    const params = [
      localPath,
      Math.round(pyFeat.size_kb),
      pyFeat.width,
      pyFeat.height,
      pyFeat.avg_r,
      pyFeat.avg_g,
      pyFeat.avg_b
    ];
    await pool.query(insertSQL, params);
  } catch (dbErr) {
    // In dev/local you can ignore DB errors; in prod you might want to alert
    console.error('[/upload] DB insert error:', dbErr);
    // optionally: if (process.env.NODE_ENV === 'production') { /* notify Sentry, etc */ }
  }

  // 5. Send back both the URL and everything from Python
  res.json({
    success:  true,
    imageUrl: localPath,
    label:    pythonLabel,
    features: pyFeat
  });
});


app.listen(PORT, () =>
  console.log(`Server running on http://localhost:${PORT}`)
);
// ------------------------------------------------------------------
