
// server.js ---------------------------------------------------------
const express  = require('express');
const multer   = require('multer');
const sharp    = require('sharp');
const path     = require('path');
const fs       = require('fs');
const fetch    = require('node-fetch');    // npm i node-fetch@2
const FormData = require('form-data');     // npm i form-data
const { Pool } = require('pg');            // npm i pg

// ── CONFIG ────────────────────────────────────────────────────────
const FLASK_URL    = `http://${process.env.FLASK_HOST || 'localhost'}:${process.env.FLASK_PORT || '5000'}`;
const PORT         = process.env.PORT || 3000;
const FRONTEND_DIR = path.join(__dirname, '..', 'FrontEnd');
const UPLOADS_DIR  = path.join(__dirname, 'uploads');


console.log('❯ DATABASE_URL=', process.env.DATABASE_URL);

// Postgres pool (will pick up PGHOST, PGUSER, PGPASSWORD, PGDATABASE, PGPORT)

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

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
    // 1. Local feature extraction
    const imagePath = path.join(UPLOADS_DIR, req.file.filename);
    const metadata  = await sharp(imagePath).metadata();
    const stats     = fs.statSync(imagePath);

    // 2. Send image to Flask for classification
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
    const { label } = await flaskResp.json();  // e.g. { label: "plastic" }

    // 3. Build features object
    const features = {
      path:      `/uploads/${req.file.filename}`,
      sizeKB:    parseFloat((stats.size / 1024).toFixed(1)),
      width:     metadata.width,
      height:    metadata.height,
      format:    metadata.format,
      label      // from Flask
    };

    // 4. Insert into Postgres
    const insertSQL = `
      INSERT INTO public.image_features
        (path, file_size_kb, width, height, format, label)
      VALUES
        ($1, $2, $3, $4, $5, $6)
    `;
    const params = [
      features.path,
      features.sizeKB,
      features.width,
      features.height,
      features.format,
      features.label
    ];
    await pool.query(insertSQL, params);

    // 5. Send response
    res.json({
      success:  true,
      imageUrl: features.path,
      features
    });

  } catch (err) {
    console.error(err);
    res.status(502).json({ success: false, error: 'Upload, classification, or database insert failed' });
  }
});

app.listen(PORT, () =>
  console.log(`Server running on http://localhost:${PORT}`)
);
// ------------------------------------------------------------------
