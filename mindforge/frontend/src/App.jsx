import React, { useState } from 'react';
import TextInputPanel from './components/TextInputPanel';
import MindMapCanvas from './components/MindMapCanvas';
import ExportBar from './components/ExportBar';
import QualityScorePanel from './components/QualityScorePanel';
import ClarificationPanel from './components/ClarificationPanel';
import SummaryPanel from './components/SummaryPanel';
import RefinementPanel from './components/RefinementPanel';
import PipelineStatusPanel from './components/PipelineStatusPanel';
import { generateMindMap, getClarificationQuestions, refineMindMap } from './services/apiService';

function App() {
  const [mindMapData, setMindMapData] = useState(null);
  const [miroJson, setMiroJson] = useState(null);
  const [testerReport, setTesterReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [step, setStep] = useState('input'); // input, clarify, result
  const [initialText, setInitialText] = useState('');
  const [questions, setQuestions] = useState([]);
  const [refinementUsed, setRefinementUsed] = useState(false);

  const handleInitialSubmit = async (text) => {
    setLoading(true);
    setError(null);
    setInitialText(text);
    try {
      const data = await getClarificationQuestions(text);
      setQuestions(data.questions);
      setStep('clarify');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze input. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClarificationSubmit = async (answers, additionalInfo) => {
    setLoading(true);
    setError(null);
    try {
      const result = await generateMindMap(initialText, answers, additionalInfo);
      setMindMapData(result.mind_map);
      setMiroJson(result.miro_json);
      setTesterReport(result.tester_report);
      setStep('result');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate mind map. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setStep('input');
    setMindMapData(null);
    setInitialText('');
    setRefinementUsed(false);
  };

  const handleRefine = async (feedback) => {
    if (refinementUsed) return;
    setLoading(true);
    setError(null);
    try {
      const result = await refineMindMap(mindMapData, feedback);
      setMindMapData(result.mind_map);
      setMiroJson(result.miro_json);
      setTesterReport(result.tester_report);
      setRefinementUsed(true);
    } catch (err) {
      setError(err.response?.data?.detail || 'Refinement failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-slate-900 text-slate-200">
      {step === 'input' && (
        <TextInputPanel onGenerate={handleInitialSubmit} loading={loading} />
      )}
      
      <main className="flex-grow p-8 flex flex-col relative overflow-hidden">
        {error && (
          <div className="absolute top-8 left-1/2 -translate-x-1/2 z-50 bg-red-500/90 text-white px-6 py-3 rounded-lg shadow-xl backdrop-blur-md border border-red-400">
            {error}
          </div>
        )}

        {step === 'input' && (
          <div className="flex-grow flex flex-col items-center justify-center text-center opacity-60">
            <div className="w-24 h-24 bg-brand-gold/10 rounded-full flex items-center justify-center mb-6">
              <span className="text-5xl">🧠</span>
            </div>
            <h2 className="text-3xl font-bold text-slate-300 mb-2">Ready to Forge?</h2>
            <p className="text-xl text-slate-500">Enter your topic on the left to start the multi-agent pipeline</p>
          </div>
        )}

        {step === 'clarify' && (
          <ClarificationPanel 
            questions={questions} 
            onSubmit={handleClarificationSubmit} 
            loading={loading} 
          />
        )}

        {step === 'result' && mindMapData && (
          <div className="flex-grow flex flex-col min-h-0 animate-in fade-in duration-700">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-slate-100 truncate pr-4">{mindMapData.title}</h2>
              <button 
                onClick={reset}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm font-medium border border-slate-700 transition-colors"
              >
                Create New Map
              </button>
            </div>
            
            <div className="flex-grow flex gap-6 min-h-0">
              <div className="flex-grow flex flex-col gap-6 overflow-y-auto pr-2 custom-scrollbar">
                <div id="mindmap-export-area" className="bg-slate-900 rounded-2xl border border-slate-700 shadow-2xl overflow-hidden relative min-h-[600px]">
                  <MindMapCanvas data={mindMapData} />
                </div>
                
                <SummaryPanel 
                  summary={mindMapData.summary} 
                  flowExplanation={mindMapData.flow_explanation} 
                />
              </div>
              
              <div className="w-80 flex flex-col gap-6">
                <QualityScorePanel report={testerReport} />
                <RefinementPanel 
                  onRefine={handleRefine} 
                  loading={loading} 
                  disabled={refinementUsed} 
                />
                <PipelineStatusPanel />
              </div>
            </div>

            <ExportBar mindMapData={mindMapData} miroJson={miroJson} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
