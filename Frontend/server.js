import express from 'express';
import cors from 'cors';
import multer from 'multer';

const upload = multer({
  storage: multer.memoryStorage(),
});

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

app.get('/api/experiments', (req, res) => {
  res.json([
    { name: 'Experiment A', accuracy: 0.87, loss: 0.23, date: '2025-01-01' },
    { name: 'Baseline Model', accuracy: 0.85, loss: 0.31, date: '2024-12-15' },
    { name: 'V2 Enhancement', accuracy: 0.92, loss: 0.18, date: '2024-11-30' },
  ]);
});

app.get('/api/metrics', (req, res) => {
  res.json({
    accuracy: 0.925,
    sensitivity: 0.918,
    specificity: 0.882,
    precision: 0.887,
    recall: 0.916,
    f1: 0.902,
  });
});

app.post('/api/predict', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: 'No image file provided',
      });
    }

    const formData = new FormData();

    const blob = new Blob(
      [req.file.buffer],
      { type: req.file.mimetype }
    );

    formData.append('image', blob, req.file.originalname);

    const response = await fetch(
      'http://localhost:5000/api/predict',
      {
        method: 'POST',
        body: formData,
      }
    );

    const result = await response.json();

    if (!response.ok) {
      return res.status(response.status).json(result);
    }

    res.json(result);

  } catch (error) {
    console.error('ML backend error:', error);

    res.status(500).json({
      error: 'Failed to connect to ML backend',
      details: error.message,
    });
  }
});

app.get('/api/reports', (req, res) => {
  const reports = [];

  for (let i = 4; i >= 1; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);

    reports.push({
      id: 'report-2025-' + i,
      type: i === 1 ? 'normal' : i === 2 ? 'ckd' : 'tumor',
      confidence: Math.floor(Math.random() * 40) + 60,
      timestamp: date.toISOString(),
      filename: 'report_2025-01-' + i + '.pdf',
    });
  }

  res.json(reports);
});

app.listen(PORT, () => {
  console.log('KidneyAI Backend API running on port ' + PORT);
});