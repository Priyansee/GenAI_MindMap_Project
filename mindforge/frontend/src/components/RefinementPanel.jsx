import React, { useState } from 'react';
import { Send, RefreshCw, Info } from 'lucide-react';

const RefinementPanel = ({ onRefine, loading, disabled }) => {
  const [feedback, setFeedback] = useState('');

  const handleSubmit = () => {
    if (feedback.trim() && !disabled) {
      onRefine(feedback);
      setFeedback('');
    }
  };

  return (
    <div className={`bg-slate-800/40 backdrop-blur-md border border-slate-700/50 rounded-2xl p-6 shadow-2xl transition-all ${disabled ? 'opacity-50 grayscale' : ''}`}>
      <div className="flex items-center gap-2 mb-4">
        <RefreshCw className={`text-brand-gold ${loading ? 'animate-spin' : ''}`} size={18} />
        <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-400">Refine Map (1 Trial)</h3>
      </div>

      {disabled ? (
        <div className="flex items-center gap-3 p-4 bg-slate-900/50 rounded-xl border border-slate-700/30 text-slate-400">
          <Info size={18} className="text-brand-gold shrink-0" />
          <p className="text-sm">Refinement trial already used for this session.</p>
        </div>
      ) : (
        <div className="space-y-4">
          <textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="e.g., 'Add a section about marketing' or 'Make it more professional'..."
            className="w-full h-24 bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-brand-gold/50 transition-colors resize-none"
          />
          
          <button
            onClick={handleSubmit}
            disabled={loading || !feedback.trim()}
            className="w-full flex items-center justify-center gap-2 py-3 bg-brand-gold/10 hover:bg-brand-gold/20 disabled:opacity-50 disabled:hover:bg-brand-gold/10 text-brand-gold font-bold rounded-xl border border-brand-gold/30 transition-all group"
          >
            {loading ? 'Refining...' : 'Apply Changes'}
            {!loading && <Send size={16} className="group-hover:translate-x-1 transition-transform" />}
          </button>
        </div>
      )}
    </div>
  );
};

export default RefinementPanel;
