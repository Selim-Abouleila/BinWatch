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
+const FLASK_URL = `http://${process.env.FLASK_HOST || 'localhost'}:${process.env.FLASK_PORT || '5000'}`;
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
  try {
    /* ---------- 1. Local feature extraction ---------------------- */
    const imagePath = path.join(UPLOADS_DIR, req.file.filename);
    const metadata  = await sharp(imagePath).metadata();
    const stats     = fs.statSync(imagePath);

    /* ---------- 2. Send image to Flask --------------------------- */
    const form = new FormData();
    form.append('image', fs.createReadStream(imagePath));

    const flaskResp = await fetch(`${FLASK_URL}/classify`, {
      method: 'POST',
      body:   form,
      headers: form.getHeaders(),   // critical for multipart/form-data
      timeout: 5000                 // ms – guards against hung Flask
    });

    if (!flaskResp.ok) {
      throw new Error(`Flask responded ${flaskResp.status}`);
    }
    const { label } = await flaskResp.json();   // e.g. { label: "plastic" }

    /* ---------- 3. Merge results & reply ------------------------- */
    const features = {
      filename: req.file.filename,
      sizeKB:   (stats.size / 1024).toFixed(1),
      width:    metadata.width,
      height:   metadata.height,
      format:   metadata.format,
      label     // from Flask
    };

    res.json({
      success:  true,
      imageUrl: `/uploads/${req.file.filename}`,
      features
    });

  } catch (err) {
    console.error(err);
    res.status(502).json({ success: false, error: 'Upload or classify failed' });
  }
});

app.listen(PORT, () =>
  console.log(`Server running on http://localhost:${PORT}`)
);
// ------------------------------------------------------------------
