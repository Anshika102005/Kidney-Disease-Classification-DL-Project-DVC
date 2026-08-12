import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HiUpload, HiCheck, HiX, HiRefresh, HiCloudUpload, HiDocumentText, HiChartBar, HiTrendingUp, HiInformationCircle, HiPhotograph, HiShieldCheck, HiExclamation, HiHeart } from 'react-icons/hi';
import { FaLayerGroup, FaHeartbeat, FaBrain, FaUserMd } from 'react-icons/fa';
import { ImSpinner8 } from 'react-icons/im';

const GlassCard = ({ children, className = '' }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5 }}
    whileHover={{ boxShadow: '0 0 30px rgba(59, 130, 246, 0.15)' }}
    className={`bg-gray-900/80 border border-gray-700/50 rounded-2xl shadow-xl p-6 ${className}`}
  >
    {children}
  </motion.div>
);

const StatusIndicator = ({ status }) => {
  const statusConfig = {
    idle: { color: 'text-gray-400', bg: 'bg-gray-800', label: 'Idle' },
    uploading: { color: 'text-blue-400', bg: 'bg-blue-900/50', label: 'Uploading' },
    predicting: { color: 'text-yellow-400', bg: 'bg-yellow-900/50', label: 'Predicting' },
    success: { color: 'text-green-400', bg: 'bg-green-900/50', label: 'Complete' },
    error: { color: 'text-red-400', bg: 'bg-red-900/50', label: 'Error' },
  };
  const cfg = statusConfig[status] || statusConfig.idle;

  return (
    <motion.div
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${cfg.bg} border border-gray-700/50`}
    >
      {status === 'predicting' ? (
        <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}>
          <ImSpinner8 className={`w-4 h-4 ${cfg.color}`} />
        </motion.div>
      ) : status === 'success' ? (
        <HiCheck className={`w-4 h-4 ${cfg.color}`} />
      ) : status === 'error' ? (
        <HiX className={`w-4 h-4 ${cfg.color}`} />
      ) : (
        <div className={`w-2 h-2 rounded-full ${cfg.color.replace('text-', 'bg-')}`} />
      )}
      <span className={`text-sm font-medium ${cfg.color}`}>{cfg.label}</span>
    </motion.div>
  );
};

const SidebarItem = ({ icon, label, active, onClick, badge }) => (
  <motion.button
    onClick={onClick}
    whileHover={{ x: 5, backgroundColor: 'rgba(59, 130, 246, 0.1)' }}
    whileTap={{ scale: 0.95 }}
    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors text-left ${
      active ? 'bg-blue-900/30 text-blue-400 border-r-2 border-blue-400' : 'text-gray-400 hover:text-gray-200'
    }`}
  >
    <span className="text-lg">{icon}</span>
    <span className="font-medium flex-1">{label}</span>
    {badge && <span className="bg-blue-500 text-white text-xs px-2 py-0.5 rounded-full">{badge}</span>}
  </motion.button>
);

const precautions = {
  normal: [
    'Continue regular health checkups annually',
    'Stay hydrated with 8-10 glasses of water daily',
    'Maintain a balanced diet rich in fruits and vegetables',
    'Exercise regularly to maintain overall health',
    'Avoid excessive salt and processed foods',
  ],
  ckd: [
    'Consult a nephrologist immediately for further evaluation',
    'Monitor blood pressure and blood sugar regularly',
    'Reduce salt and protein intake as advised by your doctor',
    'Avoid NSAIDs and nephrotoxic medications',
    'Stay hydrated but follow fluid restrictions if prescribed',
  ],
  tumor: [
    'Seek immediate consultation with a urologist or oncologist',
    'Avoid smoking and limit alcohol consumption',
    'Maintain a healthy weight through diet and exercise',
    'Follow up with imaging studies as recommended',
    'Consider genetic counseling if there is family history',
  ],
  cyst: [
    'Most kidney cysts are benign and require no treatment',
    'Monitor with periodic ultrasounds if advised by doctor',
    'Stay hydrated and avoid excessive salt intake',
    'Report any pain or changes to your healthcare provider',
    'Maintain healthy blood pressure levels',
  ],
  stone: [
    'Increase water intake to at least 3 liters daily',
    'Reduce sodium and animal protein consumption',
    'Take prescribed medications to help pass the stone',
    'Avoid foods high in oxalates if recommended',
    'Follow up with imaging to confirm stone passage',
  ],
};

const causes = [
  'High blood pressure and diabetes',
  'Smoking and excessive alcohol use',
  'Obesity and sedentary lifestyle',
  'Family history of kidney disease',
  'Prolonged use of painkillers or NSAIDs',
  'Exposure to toxic chemicals',
];

const lifestyleTips = [
  'Drink at least 8-10 glasses of water daily',
  'Reduce sodium intake to less than 2,300mg per day',
  'Eat more fresh fruits, vegetables, and whole grains',
  'Exercise for at least 30 minutes daily',
  'Get adequate sleep (7-9 hours per night)',
  'Avoid smoking and limit alcohol consumption',
  'Manage stress through meditation or yoga',
  'Regular health screenings and checkups',
];

const labelMap = {
  normal: { color: 'green', label: 'Healthy State', desc: 'Normal kidney function. No abnormalities detected.' },
  ckd: { color: 'yellow', label: 'Chronic Kidney Disease', desc: 'Signs of chronic kidney disease detected. Consult a nephrologist for further evaluation.' },
  tumor: { color: 'red', label: 'Tumor Detected', desc: 'Tumor abnormality detected. Immediate consultation with an oncologist/urologist is recommended.' },
  cyst: { color: 'blue', label: 'Cyst Detected', desc: 'Kidney cyst identified. Most cysts are benign, but consult a urologist for evaluation.' },
  stone: { color: 'yellow', label: 'Kidney Stone', desc: 'Kidney stone detected. Increase fluid intake and consult a specialist for treatment options.' },
};

const PredictionResult = ({ prediction, confidence, uploadedImageUrl, uploadedImage, onReset, error }) => {
  const info = labelMap[prediction?.toLowerCase() as keyof typeof labelMap] || labelMap.normal;
  const colorClass = info.color === 'green' ? 'green' : info.color === 'yellow' ? 'yellow' : info.color === 'blue' ? 'blue' : 'red';

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      {error && (
        <div className="p-4 rounded-lg bg-red-900/20 border border-red-700 text-red-400 text-sm">
          {error}
        </div>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <GlassCard>
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2"><HiPhotograph className="text-blue-400" /> Uploaded Scan</h3>
          <img src={uploadedImageUrl} alt="Uploaded scan" className="w-full h-64 object-cover rounded-lg border border-gray-700" />
        </GlassCard>

        <GlassCard>
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2"><FaHeartbeat className="text-red-400" /> Prediction Result</h3>
          <div className="space-y-4">
            <div className={`p-4 rounded-lg border ${colorClass === 'green' ? 'bg-green-900/20 border-green-700' : colorClass === 'yellow' ? 'bg-yellow-900/20 border-yellow-700' : colorClass === 'blue' ? 'bg-blue-900/20 border-blue-700' : 'bg-red-900/20 border-red-700'}`}>
              <div className="flex items-center justify-between mb-2">
                <span className={`text-2xl font-bold ${colorClass === 'green' ? 'text-green-400' : colorClass === 'yellow' ? 'text-yellow-400' : colorClass === 'blue' ? 'text-blue-400' : 'text-red-400'}`}>
                  {info.label}
                </span>
                <span className="text-3xl font-bold text-blue-400">{confidence.toFixed ? confidence.toFixed(2) : confidence}%</span>
              </div>
              <p className="text-sm text-gray-400">Confidence Score</p>
              <div className="mt-3 h-2 bg-gray-800 rounded-full overflow-hidden">
                <motion.div initial={{ width: 0 }} animate={{ width: `${Math.min(confidence, 100)}%` }} transition={{ duration: 1, delay: 0.3 }} className={`h-full rounded-full ${colorClass === 'green' ? 'bg-green-500' : colorClass === 'yellow' ? 'bg-yellow-500' : colorClass === 'blue' ? 'bg-blue-500' : 'bg-red-500'}`} />
              </div>
            </div>

            <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
              <p className="text-sm text-gray-400 mb-1">Analysis</p>
              <p className="text-gray-200">{info.desc}</p>
            </div>

            <div className="flex gap-3">
              <button onClick={onReset} className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center justify-center gap-2 transition-colors">
                <HiRefresh className="w-4 h-4" /> New Scan
              </button>
            </div>
          </div>
        </GlassCard>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <GlassCard>
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-yellow-400"><HiShieldCheck className="text-yellow-400" /> Precautions</h3>
          <ul className="space-y-3">
            {(precautions[prediction.toLowerCase()] || precautions.normal).map((item, index) => (
              <motion.li key={index} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 * index }} className="flex items-start gap-2 text-sm text-gray-300">
                <span className="text-yellow-400 mt-0.5">•</span>
                {item}
              </motion.li>
            ))}
          </ul>
        </GlassCard>

        <GlassCard>
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-red-400"><HiExclamation className="text-red-400" /> Causes of Kidney Tumor</h3>
          <ul className="space-y-3">
            {causes.map((item, index) => (
              <motion.li key={index} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 * index }} className="flex items-start gap-2 text-sm text-gray-300">
                <span className="text-red-400 mt-0.5">•</span>
                {item}
              </motion.li>
            ))}
          </ul>
        </GlassCard>

        <GlassCard>
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-green-400"><HiHeart className="text-green-400" /> Healthy Lifestyle</h3>
          <ul className="space-y-3">
            {lifestyleTips.map((item, index) => (
              <motion.li key={index} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 * index }} className="flex items-start gap-2 text-sm text-gray-300">
                <span className="text-green-400 mt-0.5">•</span>
                {item}
              </motion.li>
            ))}
          </ul>
        </GlassCard>
      </div>
    </motion.div>
  );
};

export default function App() {
  const [currentView, setCurrentView] = useState('predict');
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [status, setStatus] = useState('idle');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const PREDICTION_API = 'http://localhost:5000/api/predict';

  const handleImageUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setStatus('uploading');
    setUploadedImage(file);
    setUploadedImageUrl(URL.createObjectURL(file));
    setError('');

    const formData = new FormData();
    formData.append('image', file);

    try {
      setStatus('predicting');
      const response = await fetch(PREDICTION_API, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction request failed');
      }

      const result = await response.json();
      setPrediction(result.prediction);
      setConfidence(result.confidence);
      setStatus('success');
    } catch (error) {
      console.error('Prediction failed:', error);
      setStatus('error');
      setError(error.message || 'Failed to get prediction');
      setPrediction(null);
      setConfidence(0);
    }
  };

  const resetPrediction = () => {
    setPrediction(null);
    setConfidence(0);
    setStatus('idle');
    setUploadedImage(null);
    setUploadedImageUrl(null);
    setError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const experiments = [
    { name: 'Experiment A', accuracy: 87, loss: 0.23, date: '2025-01-01' },
    { name: 'Baseline Model', accuracy: 85, loss: 0.31, date: '2024-12-15' },
    { name: 'V2 Enhancement', accuracy: 92, loss: 0.18, date: '2024-11-30' },
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 overflow-hidden font-sans">
      <div className="flex h-screen">
        <motion.aside
          initial={{ x: -300, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className="w-64 bg-gray-900/95 border-r border-gray-800/50 flex flex-col p-4 fixed h-full z-50"
        >
          <div className="flex items-center gap-3 mb-8 mt-2">
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
              className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center"
            >
              <FaHeartbeat className="text-white text-xl" />
            </motion.div>
            <div>
               <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">NephroScan AI</h1>
              <p className="text-xs text-gray-500">AI-Powered Kidney CT Scan Classification</p>
            </div>
          </div>

          <nav className="flex-1 space-y-2">
            <SidebarItem icon={<FaBrain />} label="Predict" active={currentView === 'predict'} onClick={() => setCurrentView('predict')} />
            <SidebarItem icon={<FaLayerGroup />} label="Experiments" active={currentView === 'experiments'} onClick={() => setCurrentView('experiments')} />
            <SidebarItem icon={<HiDocumentText />} label="Reports" active={currentView === 'reports'} onClick={() => setCurrentView('reports')} badge="New" />
            <SidebarItem icon={<HiChartBar />} label="Analytics" active={currentView === 'analytics'} onClick={() => setCurrentView('analytics')} />
            <SidebarItem icon={<HiShieldCheck />} label="Precautions" active={currentView === 'precautions'} onClick={() => setCurrentView('precautions')} />
            <SidebarItem icon={<HiHeart />} label="Lifestyle" active={currentView === 'lifestyle'} onClick={() => setCurrentView('lifestyle')} />
          </nav>

          <div className="mt-auto pt-4 border-t border-gray-800">
            <div className="p-4 rounded-lg bg-gradient-to-r from-blue-900/30 to-purple-900/30 border border-blue-500/30">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
                  <FaUserMd className="text-white text-lg" />
                </div>
                <div>
                  <p className="font-semibold text-sm">Dr. AI Assistant</p>
                  <p className="text-xs text-gray-400">Online</p>
                </div>
              </div>
            </div>
          </div>
        </motion.aside>

        <main className="flex-1 ml-64 p-6 lg:p-8 overflow-y-auto h-screen">
          <AnimatePresence mode="wait">
            {currentView === 'predict' && (
              <motion.div key="predict" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }} className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-3xl font-bold mb-2">Kidney Disease Classification</h2>
                    <p className="text-gray-400">Upload a kidney scan for AI-powered disease prediction</p>
                  </div>
                  <div className="hidden lg:flex items-center gap-4">
                    <StatusIndicator status={status} />
                    {prediction && (
                      <motion.button whileHover={{ scale: 1.05, rotate: 180 }} whileTap={{ scale: 0.95 }} onClick={resetPrediction} className="p-2 rounded-full bg-gray-800 hover:bg-gray-700 transition-colors">
                        <HiRefresh className="w-5 h-5 text-gray-400" />
                      </motion.button>
                    )}
                  </div>
                </div>

                 {!prediction ? (
                  <GlassCard>
                    <div className="mb-4">
                      <h3 className="text-xl font-semibold mb-2 flex items-center gap-2"><HiPhotograph className="text-blue-400" /> Upload Medical Scan</h3>
                      <p className="text-gray-400 text-sm">Supported: PNG, JPG, JPEG (Recommended 1024x1024)</p>
                    </div>
                    <label className="block w-full border-2 border-dashed border-gray-600 hover:border-blue-500 hover:bg-blue-500/5 rounded-xl p-12 text-center cursor-pointer transition-all">
                      <input ref={fileInputRef} type="file" accept=".png,.jpg,.jpeg" onChange={handleImageUpload} className="hidden" />
                      <div className="space-y-4">
                        <motion.div animate={{ y: [0, -10, 0] }} transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}>
                          <HiUpload className="w-16 h-16 text-gray-400 mx-auto" />
                        </motion.div>
                        <div>
                          <p className="text-lg font-semibold text-gray-200">Drop your scan here or click to browse</p>
                          <p className="text-sm text-gray-500">PNG, JPG up to 10MB</p>
                        </div>
                      </div>
                    </label>
                  </GlassCard>
                ) : (
                  <PredictionResult
                    prediction={prediction}
                    confidence={confidence}
                    uploadedImageUrl={uploadedImageUrl}
                    uploadedImage={uploadedImage}
                    onReset={resetPrediction}
                    error={error}
                  />
                )}

                <GlassCard className="mt-6">
                  <h3 className="text-xl font-semibold mb-2 flex items-center gap-2"><HiInformationCircle className="text-blue-400" /> About NephroScan AI</h3>
                  <p className="text-sm text-gray-300 leading-relaxed">
                    NephroScan AI is a Deep Learning-based Kidney CT Scan Classification system that analyzes
                    medical CT images and classifies them into four categories: Normal, Cyst, Stone, and Tumor.
                    The application uses a VGG16 transfer learning model with TensorFlow and Keras, while Flask
                    powers the backend API and React with Tailwind CSS provides the user interface. It delivers
                    fast, accurate predictions along with confidence scores to assist in AI-based kidney disease
                    analysis.
                  </p>
                </GlassCard>
              </motion.div>
            )}

            {currentView === 'experiments' && (
              <motion.div key="experiments" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }}>
                <GlassCard>
                  <h3 className="text-xl font-semibold mb-6 flex items-center gap-2"><FaLayerGroup className="text-blue-400" /> MLflow Experiments</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {experiments.map((exp, index) => (
                      <motion.div key={exp.name} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 * (index + 1) }} whileHover={{ scale: 1.03, y: -5 }} className="p-4 rounded-lg bg-gray-800/50 border border-gray-700 hover:border-blue-500/30 transition-all cursor-pointer">
                        <h4 className="font-semibold text-lg mb-2">{exp.name}</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between"><span className="text-gray-400">Accuracy:</span><span className="text-green-400 font-semibold">{exp.accuracy}%</span></div>
                          <div className="flex justify-between"><span className="text-gray-400">Loss:</span><span className="text-red-400">{exp.loss}</span></div>
                          <div className="flex justify-between"><span className="text-gray-400">Date:</span><span>{exp.date}</span></div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </GlassCard>
              </motion.div>
            )}

            {currentView === 'reports' && (
              <motion.div key="reports" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }} className="space-y-6">
                <GlassCard>
                  <h3 className="text-xl font-semibold mb-6 flex items-center gap-2"><HiDocumentText className="text-purple-400" /> Recent Reports</h3>
                  <div className="space-y-4">
                    {[4, 3, 2, 1].map((i) => (
                      <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 * i }} whileHover={{ x: 5 }} className="p-4 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors cursor-pointer">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4">
                            <div className="w-10 h-10 rounded-full bg-blue-900/30 flex items-center justify-center">
                              <HiDocumentText className="text-blue-400" />
                            </div>
                            <div>
                              <p className="font-semibold">Scan Report #202{i}</p>
                              <p className="text-xs text-gray-500">Generated: Jan 2{i + 1}, 2025</p>
                            </div>
                          </div>
                          <HiDocumentText className="w-5 h-5 text-gray-500" />
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </GlassCard>
              </motion.div>
            )}

            {currentView === 'analytics' && (
              <motion.div key="analytics" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }} className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <GlassCard>
                    <h3 className="text-xl font-semibold mb-6 flex items-center gap-2"><HiChartBar className="text-green-400" /> Prediction Distribution</h3>
                    <div className="flex items-center gap-8">
                      <div className="relative w-48 h-48 flex-shrink-0">
                        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                          <circle cx="50" cy="50" r="40" fill="none" stroke="#10b981" strokeWidth="20" strokeDasharray="150.8 251.3" strokeDashoffset="0" />
                          <circle cx="50" cy="50" r="40" fill="none" stroke="#f59e0b" strokeWidth="20" strokeDasharray="75.4 251.3" strokeDashoffset="-150.8" />
                          <circle cx="50" cy="50" r="40" fill="none" stroke="#ef4444" strokeWidth="20" strokeDasharray="25.1 251.3" strokeDashoffset="-226.2" />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="text-center">
                            <p className="text-2xl font-bold">85</p>
                            <p className="text-xs text-gray-400">Total</p>
                          </div>
                        </div>
                      </div>
                      <div className="space-y-3">
                        {[
                          { label: 'Normal', color: 'bg-green-500', pct: '60%' },
                          { label: 'CKD', color: 'bg-yellow-500', pct: '30%' },
                          { label: 'Tumor', color: 'bg-red-500', pct: '10%' },
                        ].map((item) => (
                          <div key={item.label} className="flex items-center gap-3">
                            <div className={`w-3 h-3 ${item.color} rounded-full`} />
                            <span className="text-sm text-gray-300">{item.label}</span>
                            <span className="text-sm font-semibold ml-auto">{item.pct}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </GlassCard>

                  <GlassCard>
                    <h3 className="text-xl font-semibold mb-6 flex items-center gap-2"><FaBrain className="text-purple-400" /> Model Metrics</h3>
                    <div className="space-y-6">
                      {[
                        { label: 'Accuracy', value: 92.5 },
                        { label: 'Sensitivity', value: 91.8 },
                        { label: 'Specificity', value: 88.2 },
                      ].map((m, i) => (
                        <div key={m.label}>
                          <div className="flex justify-between text-sm mb-2">
                            <span className="text-gray-400">{m.label}</span>
                            <span className="font-semibold">{m.value}%</span>
                          </div>
                          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              whileInView={{ width: `${m.value}%` }}
                              transition={{ duration: 1, delay: 0.5 + i * 0.3 }}
                              viewport={{ once: true }}
                              className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </GlassCard>
                </div>
              </motion.div>
            )}

            {currentView === 'precautions' && (
              <motion.div key="precautions" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }} className="space-y-6">
                <GlassCard>
                  <h3 className="text-xl font-semibold mb-6 flex items-center gap-2"><HiShieldCheck className="text-yellow-400" /> Precautions for Kidney Health</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {Object.entries(precautions).map(([key, items]) => (
                      <div key={key} className={`p-4 rounded-lg border ${key === 'normal' ? 'bg-green-900/20 border-green-700' : key === 'cyst' ? 'bg-blue-900/20 border-blue-700' : key === 'stone' ? 'bg-yellow-900/20 border-yellow-700' : 'bg-red-900/20 border-red-700'}`}>
                        <h4 className="font-semibold mb-3 capitalize">{key === 'normal' ? 'Healthy' : key === 'cyst' ? 'Cyst' : key === 'stone' ? 'Stone' : 'Tumor'}</h4>
                        <ul className="space-y-2">
                          {items.map((item, index) => (
                            <li key={index} className="flex items-start gap-2 text-sm text-gray-300">
                              <span className={`mt-0.5 ${key === 'normal' ? 'text-green-400' : key === 'cyst' ? 'text-blue-400' : key === 'stone' ? 'text-yellow-400' : 'text-red-400'}`}>•</span>
                              {item}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                </GlassCard>
              </motion.div>
            )}

            {currentView === 'lifestyle' && (
              <motion.div key="lifestyle" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }} className="space-y-6">
                <GlassCard>
                  <h3 className="text-xl font-semibold mb-6 flex items-center gap-2"><HiHeart className="text-green-400" /> Healthy Lifestyle for Kidney Care</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {lifestyleTips.map((tip, index) => (
                      <motion.div key={index} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 * index }} className="p-4 rounded-lg bg-gray-800/50 border border-gray-700 hover:border-green-500/30 transition-all">
                        <div className="flex items-start gap-3">
                          <span className="text-green-400 text-lg">•</span>
                          <p className="text-sm text-gray-300">{tip}</p>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </GlassCard>

                <GlassCard>
                  <h3 className="text-xl font-semibold mb-6 flex items-center gap-2"><HiExclamation className="text-red-400" /> Causes of Kidney Tumor</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {causes.map((cause, index) => (
                      <motion.div key={index} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 * index }} className="p-4 rounded-lg bg-gray-800/50 border border-gray-700 hover:border-red-500/30 transition-all flex items-center gap-3">
                        <span className="text-red-400 text-lg">•</span>
                        <p className="text-sm text-gray-300">{cause}</p>
                      </motion.div>
                    ))}
                  </div>
                </GlassCard>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
