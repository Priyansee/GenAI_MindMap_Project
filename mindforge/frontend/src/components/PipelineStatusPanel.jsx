import React from 'react';
import { Cpu, Layers, Share2, Activity } from 'lucide-react';

const PipelineStatusPanel = () => {
  return (
    <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 rounded-2xl p-6 shadow-2xl relative overflow-hidden group">
      <div className="flex items-center gap-2 mb-6">
        <Activity className="text-blue-400" size={18} />
        <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-400">Core Pipeline</h3>
      </div>

      <div className="space-y-6">
        <StatusItem
          icon={<Cpu size={16} />}
          label="Multi-Agent"
          status="Operational"
          color="text-emerald-400"
          active
        />
        <StatusItem
          icon={<Layers size={16} />}
          label="Heuristic"
          status="Optimized"
          color="text-blue-400"
        />
        <StatusItem
          icon={<Share2 size={16} />}
          label="Relational"
          status="Verified"
          color="text-brand-gold"
        />
      </div>

      {/* Pulsing indicator */}
      <div className="mt-8 flex items-center gap-2 px-3 py-1.5 bg-slate-900/50 rounded-full border border-slate-700/50 w-fit">
        <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-tighter">System Synced</span>
      </div>
    </div>
  );
};

const StatusItem = ({ icon, label, status, color, active }) => (
  <div className="group/item flex items-center justify-between">
    <div className="flex items-center gap-3">
      <div className={`p-2 rounded-lg bg-slate-900/50 border border-slate-700/50 ${active ? 'text-brand-gold shadow-[0_0_10px_rgba(200,169,110,0.1)]' : 'text-slate-500'}`}>
        {icon}
      </div>
      <div className="flex flex-col">
        <span className="text-[10px] font-black uppercase tracking-widest text-slate-500">{label}</span>
        <span className={`text-xs font-mono font-bold ${color}`}>{status}</span>
      </div>
    </div>
    {active && (
      <div className="flex gap-0.5">
        {[1, 2, 3].map(i => (
          <div key={i} className="w-1 h-3 bg-brand-gold/30 rounded-full animate-bounce" style={{ animationDelay: `${i * 0.1}s` }} />
        ))}
      </div>
    )}
  </div>
);

export default PipelineStatusPanel;
