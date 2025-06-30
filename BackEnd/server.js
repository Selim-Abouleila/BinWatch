
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
const PORT         = 8080;
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
  try {
    // 1. Ensure we have a file
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No file uploaded' });
    }

    // 2. Build the local URL & filepath
    const localPath = `/uploads/${req.file.filename}`;
    const imagePath = path.join(UPLOADS_DIR, req.file.filename);

    // 3. Send the image on to Flask for all the heavy lifting
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

    // 4. Pull back the Python‐computed features + auto‐label
    //    Expected shape: { success: true, label: "pleine", features: { filename, width, height, size_kb, avg_r, avg_g, avg_b, ground_ratio } }
    const { label: pythonLabel, features: pyFeat } = await flaskResp.json();

    // 5. Insert into Postgres just the columns we have
    const insertSQL = `
      INSERT INTO public.image_features
        (path, file_size_kb, width, height, mean_r, mean_g, mean_b)
      VALUES
        ($1, $2, $3, $4, $5, $6, $7)
    `;
    const params = [
      localPath,        // maps to `path`
      Math.round(pyFeat.size_kb),  // file_size_kb  (integer)
      pyFeat.width,     // width
      pyFeat.height,    // height
      pyFeat.avg_r,     // mean_r
      pyFeat.avg_g,     // mean_g
      pyFeat.avg_b      // mean_b
    ];
    await pool.query(insertSQL, params);

    // 6. Send back both the URL and everything from Python
    res.json({
  success:  true,
  imageUrl: localPath,      // unchanged
  label:    pythonLabel,    // rename from pythonLabel → label
  features: pyFeat          // rename from pythonFeatures → features
});

  } catch (err) {
    console.error('[/upload] error:', err);
    res.status(502).json({ success: false, error: 'Upload, classification, or DB insert failed' });
  }
});


app.listen(PORT, () =>
  console.log(`Server running on http://localhost:${PORT}`)
);
// ------------------------------------------------------------------
