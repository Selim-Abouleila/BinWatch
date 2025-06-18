// database.js
const mysql = require('mysql2');
require('dotenv').config();

const pool = mysql.createPool({
  host            : process.env.DB_HOST,
  port            : process.env.DB_PORT,
  user            : process.env.DB_USER,
  password        : process.env.DB_PASSWORD,
  database        : process.env.DB_NAME,
  waitForConnections : true,
  connectionLimit    : 10,
  queueLimit         : 0
});

// Now export exactly the same interface your routes expect:
module.exports = {
  query: (...args) => pool.query(...args),
  // if you ever need to get a raw connection:
  getConnection: (...args) => pool.getConnection(...args),
};

const FLASK_HOST = process.env.FLASK_HOST;   // example: python-api.up.railway.internal
const FLASK_PORT = process.env.FLASK_PORT;   // example: 8080
const FLASK_URL  = `http://${FLASK_HOST}:${FLASK_PORT}`;  // full base URL