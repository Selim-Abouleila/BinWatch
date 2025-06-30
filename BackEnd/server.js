// server.js ---------------------------------------------------------
const express  = require('express');
const multer   = require('multer');
const sharp    = require('sharp');
const path     = require('path');
const fs       = require('fs');

// ── NEW: talk to Flask ──────────────────────────────────────────────
/*  If you run Node ≥18 you can replace the two lines below with:
 *     const fetch = global.fetch;
 *     const FormData = require('form-data');
 */
const fetch     = require('node-fetch');   // npm i node-fetch@2
const FormData  = require('form-data');    // npm i form-data
const FLASK_URL = `http://${process.env.FLASK_HOST || 'localhost'}:${process.env.FLASK_PORT || '5000'}`;
// ────────────────────────────────────────────────────────────────────

const app  = express();
const PORT = process.env.PORT || 3000;

// Paths
const FRONTEND_DIR = path.join(__dirname, '..', 'FrontEnd');
const UPLOADS_DIR  = path.join(__dirname, 'uploads');

// Ensure uploads folder exists
fs.mkdirSync(UPLOADS_DIR, { recursive: true });

// Serve your static front-end
app.use(express.static(FRONTEND_DIR));
app.use('/uploads', express.static(UPLOADS_DIR));

// Multer config
const storage = multer.diskStorage({
  destination: UPLOADS_DIR,
  filename:  (req, file, cb) => cb(null, Date.now() + '-' + file.originalname)
});
const upload = multer({ storage });

// Body parsers
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Route: serve index.html
app.get('/', (req, res) =>
  res.sendFile(path.join(FRONTEND_DIR, 'index.html'))
);

// Route: upload → local metadata → call Flask → respond
app.post('/upload', upload.single('image'), async (req, res) => {
  // Validate upload
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'No file uploaded' });
  }

  try {
    // 1. Local feature extraction
    const imagePath = path.join(UPLOADS_DIR, req.file.filename);
    const { width, height, format } = await sharp(imagePath).metadata();
    const { size } = fs.statSync(imagePath);
    const fileSizeKB = Math.round(size / 1024);

    // 2. Send image to Flask for classification
    const form = new FormData();
    form.append('image', fs.createReadStream(imagePath));

    const flaskResp = await fetch(`${FLASK_URL}/classify`, {
      method: 'POST',
      body:   form,
      headers: form.getHeaders(),
      timeout: 30000
    });

    if (!flaskResp.ok) {
      throw new Error(`Flask responded ${flaskResp.status}`);
    }

    const { label } = await flaskResp.json(); // e.g. { label: 'plastic' }

    // 3. Insert available fields into PostgreSQL
    const insertQuery = `
      INSERT INTO public.image_features
        (path, file_size_kb, width, height)
      VALUES ($1, $2, $3, $4)
      RETURNING id;
    `;

    const values = [
      `/uploads/${req.file.filename}`,
      fileSizeKB,
      width,
      height
    ];

    const dbRes = await pool.query(insertQuery, values);
    const newId = dbRes.rows[0].id;

    // 4. Reply to client with DB id and features
    return res.status(201).json({
      success:  true,
      id:       newId,
      imageUrl: `/uploads/${req.file.filename}`,
      features: {
        filename: req.file.filename,
        sizeKB:   fileSizeKB,
        width,
        height,
        format,
        label
      }
    });

  } catch (error) {
    console.error('Error in /upload:', error);
    // Cleanup on failure
    if (req.file) {
      try { fs.unlinkSync(path.join(UPLOADS_DIR, req.file.filename)); } catch {}
    }
    return res.status(500).json({ success: false, error: 'Upload, classification, or database insertion failed' });
  }
});
// ------------------------------------------------------------------
