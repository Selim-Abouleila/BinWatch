const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.static('public'));
app.use(express.urlencoded({ extended: true }));
app.set('view engine', 'ejs');

const storage = multer.diskStorage({
  destination: 'uploads/',
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ storage });

app.get('/', (req, res) => {
  res.render('index');
});

app.post('/upload', upload.single('image'), async (req, res) => {
  const imagePath = req.file.path;

  // Extract features
  const metadata = await sharp(imagePath).metadata();
  const stats = fs.statSync(imagePath);

  const features = {
    filename: req.file.filename,
    sizeKB: (stats.size / 1024).toFixed(1),
    width: metadata.width,
    height: metadata.height,
    format: metadata.format,
    avgColor: metadata.isProgressive ? 'N/A' : 'TBD'
  };

  res.render('index', { image: `/uploads/${req.file.filename}`, features });
});

app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
