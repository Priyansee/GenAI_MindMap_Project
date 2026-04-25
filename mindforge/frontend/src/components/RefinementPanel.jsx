import React, { useState } from 'react';
import { Send, RefreshCw, Info } from 'lucide-react';

const RefinementPanel = ({ onRefine, loading, disabled }) => {
  const [feedback, setFeedback] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (feedback.trim() && !disabled) {
      onRefine(feedback);
      setFeedback('');
    }
  };

  return (
    <div 
      data-html2canvas-ignore="true"
      className={`absolute bottom-6 left-1/2 -translate-x-1/2 w-[90%] max-w-2xl z-20 transition-all ${disabled ? 'opacity-50 pointer-events-none' : ''}`}
    >
      <form 
        onSubmit={handleSubmit}
        className="bg-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-2 shadow-2xl flex items-center gap-2 group focus-within:border-brand-gold/40 transition-all"
      >
        <div className="flex items-center gap-3 pl-4 pr-2 text-slate-400 shrink-0">
          <RefreshCw className={`text-brand-gold ${loading ? 'animate-spin' : ''}`} size={18} />
          <span className="text-[10px] font-black uppercase tracking-widest hidden sm:block">Regenerator</span>
        </div>

        <input
          type="text"
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          placeholder={disabled ? "One-trial limit reached" : "Enter update instructions (e.g. 'Add more details to phase 2')..."}
          disabled={disabled || loading}
          className="flex-grow bg-transparent border-none py-3 px-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-0"
        />
        
        <button
          type="submit"
          disabled={loading || !feedback.trim() || disabled}
          className="flex items-center justify-center gap-2 px-6 py-2.5 bg-brand-gold/20 hover:bg-brand-gold/30 disabled:opacity-30 text-brand-gold font-bold rounded-xl border border-brand-gold/30 transition-all active:scale-95 shrink-0"
        >
          {loading ? 'Refining...' : 'Update Map'}
          {!loading && <Send size={16} />}
        </button>
      </form>
      
      {!disabled && (
        <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-900/80 backdrop-blur px-3 py-1 rounded-full border border-slate-800 text-[9px] uppercase tracking-tighter text-slate-500 font-bold">
          One Trial Remaining
        </div>
      )}
    </div>
  );
};

export default RefinementPanel;
