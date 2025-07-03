// server.js ---------------------------------------------------------
const express  = require('express');
const multer   = require('multer');
const sharp    = require('sharp');
const path     = require('path');
const fs       = require('fs');
const fetch    = require('node-fetch');      // npm i node-fetch@2
const FormData = require('form-data');       // npm i form-data
const { Pool } = require('pg');              // npm i pg
const bcrypt   = require('bcrypt');          // npm i bcrypt
const jwt      = require('jsonwebtoken');    // npm i jsonwebtoken

console.log('❯ DATABASE_URL       =', process.env.DATABASE_URL);

// ── CONSTANTS & CONFIG ────────────────────────────────────────────
const SALT_ROUNDS = 10;
const JWT_SECRET  = process.env.JWT_SECRET || 'change_this_secret';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

const FLASK_URL    = `http://${process.env.FLASK_HOST || 'localhost'}:${process.env.FLASK_PORT || '5000'}`;
const PORT         = process.env.PORT || 3000;
const FRONTEND_DIR = path.join(__dirname, '..', 'FrontEnd');
const UPLOADS_DIR  = path.join(__dirname, 'uploads');

// Ensure uploads folder exists
fs.mkdirSync(UPLOADS_DIR, { recursive: true });

// ── APP SETUP ─────────────────────────────────────────────────────
const app = express();
app.use(express.static(FRONTEND_DIR));
app.use('/uploads', express.static(UPLOADS_DIR));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// ── MULTER CONFIG ────────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: UPLOADS_DIR,
  filename:  (req, file, cb) => cb(null, Date.now() + '-' + file.originalname)
});
const upload = multer({ storage });

// ── ROUTES ───────────────────────────────────────────────────────

// Serve SPA entrypoint
app.get('/', (req, res) =>
  res.sendFile(path.join(FRONTEND_DIR, 'index.html'))
);

// ------------------------------------------------------------------
// AUTHENTICATION ROUTES USING TABLE public.utilisateur
// ------------------------------------------------------------------

/*
Table public.utilisateur structure (expected)
-------------------------------------------------
 id             SERIAL PRIMARY KEY
 prenom         VARCHAR(100)   NOT NULL
 nom            VARCHAR(100)   NOT NULL
 email          VARCHAR(255)   UNIQUE NOT NULL
 ville          VARCHAR(100)
 mot_de_passe   TEXT           NOT NULL   -- stores *hashed* password
 date_creation  TIMESTAMP      DEFAULT CURRENT_TIMESTAMP
*/

// POST /register – create user account (ville → "confidentiel")
app.post('/register', async (req, res) => {
  // Accept ville from body; fallback to "confidentiel" if absent/empty
  let { prenom, nom, email, ville, password } = req.body;
  ville = (ville || '').trim() || 'confidentiel';

  // Basic validation (ville may be empty)
  if (!prenom || !nom || !email || !password) {
    return res.status(400).json({ success: false, error: 'Champs requis manquants' });
  }

  try {
    // Check if email already exists
    const exists = await pool.query('SELECT 1 FROM public.utilisateur WHERE email = $1', [email]);
    if (exists.rowCount > 0) {
      return res.status(409).json({ success: false, error: 'Email déjà utilisé' });
    }

    // Hash password
    const hash = await bcrypt.hash(password, SALT_ROUNDS);

    // Insert new user
    const insertSQL = `
      INSERT INTO public.utilisateur (prenom, nom, email, ville, mot_de_passe)
      VALUES ($1, $2, $3, $4, $5)
      RETURNING id
    `;
    const { rows } = await pool.query(insertSQL, [prenom, nom, email, ville, hash]);
    const userId = rows[0].id;

    // Issue JWT token
    const token = jwt.sign({ userId, email, prenom, nom }, JWT_SECRET, { expiresIn: '7d' });
    res.status(201).json({ success: true, token, user: { id: userId, prenom, nom, email, ville } });
  } catch (err) {
    console.error('[/register] error:', err);
    res.status(500).json({ success: false, error: "Échec de l'inscription" });
  }
});

// POST /login – authenticate user
app.post('/login', async (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({ success: false, error: 'Email et mot de passe requis' });
  }

  try {
    const userRes = await pool.query(
      'SELECT id, prenom, nom, mot_de_passe, ville FROM public.utilisateur WHERE email = $1',
      [email]
    );

    if (userRes.rowCount === 0) {
      return res.status(401).json({ success: false, error: 'Identifiants invalides' });
    }

    const { id, prenom, nom, mot_de_passe: hash, ville } = userRes.rows[0];
    const match = await bcrypt.compare(password, hash);

    if (!match) {
      return res.status(401).json({ success: false, error: 'Identifiants invalides' });
    }

    const token = jwt.sign({ userId: id, email, prenom, nom }, JWT_SECRET, { expiresIn: '7d' });
    res.json({ success: true, token, user: { id, prenom, nom, email, ville } });
  } catch (err) {
    console.error('[/login] error:', err);
    res.status(500).json({ success: false, error: 'Échec de connexion' });
  }
});

// ------------------------------------------------------------------
// IMAGE ROUTES (unchanged)
// ------------------------------------------------------------------

// POST /upload → classify image, store metadata, history
app.post('/upload', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'Aucun fichier téléchargé' });
  }

  const localPath  = `/uploads/${req.file.filename}`;
  const imagePath  = path.join(UPLOADS_DIR, req.file.filename);
  const { annotation, location, date } = req.body;

  let pythonLabel, pyFeat;

  // Call Flask for classification
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
    return res.status(502).json({ success: false, error: "Classification d'image impossible" });
  }

  // Insert into DB
  try {
    // 1. image_features
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

    // 2. image_history
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
    console.error('[ /upload ] DB insert error:', err);
    // Do not fail the request if DB insert fails
  }

  res.json({
    success:  true,
    imageUrl: localPath,
    label:    pythonLabel,
    features: pyFeat
  });
});

// GET /history → latest 100 images
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
    console.error('[/history] DB error:', err);
    res.status(500).json({ success: false, error: 'Erreur de lecture de la base' });
  }
});


app.get('/me/city', auth, async (req, res) => {
  try {
    const result = await pool.query(
      'SELECT ville FROM public.utilisateur WHERE id = $1',
      [req.user.userId]
    );
    const ville = result.rows[0]?.ville || null;   // "null" si non renseigné
    res.json({ success: true, ville });
  } catch (err) {
    console.error('[/me/city] error:', err);
    res.status(500).json({ success: false, error: 'Erreur serveur' });
  }
});


// ── START SERVER ─────────────────────────────────────────────────
app.listen(PORT, () =>
  console.log(`Server running on http://localhost:${PORT}`)
);
// ------------------------------------------------------------------
