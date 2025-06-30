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
  try {
    // Ensure file was uploaded
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No file uploaded' });
    }

    // 1. Local metadata extraction
    const imagePath = path.join(UPLOADS_DIR, req.file.filename);
    const metadata  = await sharp(imagePath).metadata();
    const stats     = fs.statSync(imagePath);
    const fileSizeKB = Math.round(stats.size / 1024);

    // 2. Send to Flask if needed...
    // (Omitted for this test)

    // 3. Hard-coded INSERT command (for testing)
    const testSql = `
      INSERT INTO public.image_features (
        path,
        file_size_kb,
        width,
        height,
        mean_r,
        mean_g,
        mean_b,
        luminance,
        contrast_rgb,
        contrast_gray,
        dark_pixel_ratio,
        entropy,
        std_r,
        std_g,
        std_b,
        median_r,
        median_g,
        median_b,
        min_r,
        min_g,
        min_b,
        max_r,
        max_g,
        max_b,
        hist_r,
        hist_g,
        hist_b,
        hist_luminance,
        edges
      ) VALUES (
        '/uploads/${req.file.filename}',
        ${fileSizeKB},
        ${metadata.width},
        ${metadata.height},
        123.5,
        110.2,
        98.7,
        115.4,
        20.3,
        15.1,
        0.12,
        5.6,
        25.4,
        22.1,
        18.7,
        120.0,
        100.0,
        90.0,
        0.0,
        0.0,
        0.0,
        255.0,
        240.0,
        230.0,
        ARRAY[0.1, 0.2, 0.3],
        ARRAY[0.2, 0.3, 0.4],
        ARRAY[0.15, 0.25, 0.35],
        ARRAY[0.05, 0.1, 0.15],
        ARRAY[1, 0, 1, 0, 1]
      );
    `;

    // Execute the raw SQL
    const { rows } = await pool.query(testSql);
    const newId = rows[0].id;

    // 4. Respond
    return res.status(201).json({ success: true, id: newId });
  } catch (error) {
    console.error('Test insert failed:', error);
    return res.status(500).json({ success: false, error: 'Test insert failed' });
  }
});

app.listen(PORT, () =>
  console.log(`Server running on http://localhost:${PORT}`)
);
// ------------------------------------------------------------------
