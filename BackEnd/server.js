const express = require('express');
const multer  = require('multer');
const sharp   = require('sharp');
const path    = require('path');
const fs      = require('fs');

const app  = express();
const PORT = process.env.PORT || 3000;

// Paths
const FRONTEND_DIR = path.join(__dirname, '..', 'FrontEnd');
const UPLOADS_DIR  = path.join(__dirname, 'uploads');

// Ensure uploads folder exists
fs.mkdirSync(UPLOADS_DIR, { recursive: true });

// Serve your static front-end (index.html, CSS, JS, etc.)
app.use(express.static(FRONTEND_DIR));

// Serve uploaded images
app.use('/uploads', express.static(UPLOADS_DIR));

// Multer config
const storage = multer.diskStorage({
  destination: UPLOADS_DIR,
  filename:  (req, file, cb) => cb(null, Date.now() + '-' + file.originalname)
});
const upload = multer({ storage });

// Parse form data
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Route: serve index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, 'index.html'));
});

// Route: handle upload + feature extraction
app.post('/upload', upload.single('image'), async (req, res) => {
  try {
    const imagePath = path.join(UPLOADS_DIR, req.file.filename);
    const metadata  = await sharp(imagePath).metadata();
    const stats     = fs.statSync(imagePath);

    const features = {
      filename: req.file.filename,
      sizeKB:   (stats.size / 1024).toFixed(1),
      width:    metadata.width,
      height:   metadata.height,
      format:   metadata.format,
      // you can compute avgColor here if you wish
      avgColor: 'TBD'
    };

    // Return JSON so your front-end can display it dynamically
    res.json({
      success:  true,
      imageUrl: `/uploads/${req.file.filename}`,
      features
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, error: 'Upload failed' });
  }
});

app.listen(PORT, () =>
  console.log(`Server running on http://localhost:${PORT}`)
);
