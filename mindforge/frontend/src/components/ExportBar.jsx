import React, { useState } from 'react';
import { Share2, FileJson, FileImage, FileText, ChevronUp } from 'lucide-react';
import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';

const ExportBar = ({ mindMapData, miroJson }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  const exportPNG = async () => {
    const area = document.getElementById('mindmap-export-area');
    if (!area) return;
    window.dispatchEvent(new CustomEvent('mindforge-fit-view'));
    await new Promise(r => setTimeout(r, 300));

    const canvas = await html2canvas(area, {
      scale: 3,
      useCORS: true,
      backgroundColor: '#0f172a',
      onclone: (clonedDoc) => {
        const clonedArea = clonedDoc.getElementById('mindmap-export-area');
        if (clonedArea) {
          clonedArea.style.overflow = 'visible';
          clonedArea.style.height = 'auto';
        }
      }
    });

    const link = document.createElement('a');
    link.download = `${mindMapData.title.toLowerCase().replace(/\s+/g, '-')}.png`;
    link.href = canvas.toDataURL('image/png', 1.0);
    link.click();
    setIsOpen(false);
  };

  const exportPDF = async () => {
    const area = document.getElementById('mindmap-export-area');
    if (!area) return;
    window.dispatchEvent(new CustomEvent('mindforge-fit-view'));
    await new Promise(r => setTimeout(r, 300));
    
    const canvas = await html2canvas(area, { 
      scale: 3, 
      useCORS: true, 
      backgroundColor: '#0f172a'
    });

    const imgData = canvas.toDataURL('image/png', 1.0);
    const pdf = new jsPDF('l', 'px', [canvas.width, canvas.height]);
    pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
    pdf.save(`${mindMapData.title.toLowerCase().replace(/\s+/g, '-')}.pdf`);
    setIsOpen(false);
  };

  const exportJSON = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(miroJson || mindMapData, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `${mindMapData.title.toLowerCase().replace(/\s+/g, '-')}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
    setIsOpen(false);
  };

  return (
    <div 
      data-html2canvas-ignore="true"
      className="absolute bottom-6 right-6 z-30"
    >
      <div className="relative">
        {/* Dropdown Menu */}
        {isOpen && (
          <div className="absolute bottom-full mb-3 right-0 w-56 bg-slate-800/90 backdrop-blur-xl border border-slate-700 rounded-2xl shadow-2xl overflow-hidden animate-in slide-in-from-bottom-4 duration-200">
            <button 
              onClick={exportPDF}
              className="w-full flex items-center gap-3 px-5 py-4 hover:bg-slate-700/50 text-slate-200 transition-colors border-b border-slate-700/50"
            >
              <FileText size={18} className="text-red-400" />
              <span className="text-sm font-medium">Download PDF</span>
            </button>
            <button 
              onClick={exportPNG}
              className="w-full flex items-center gap-3 px-5 py-4 hover:bg-slate-700/50 text-slate-200 transition-colors border-b border-slate-700/50"
            >
              <FileImage size={18} className="text-blue-400" />
              <span className="text-sm font-medium">Download Image</span>
            </button>
            <button 
              onClick={exportJSON}
              className="w-full flex items-center gap-3 px-5 py-4 hover:bg-slate-700/50 text-slate-200 transition-colors"
            >
              <FileJson size={18} className="text-brand-gold" />
              <span className="text-sm font-medium">Download JSON</span>
            </button>
          </div>
        )}

        {/* Main Share Button */}
        <button 
          onClick={() => setIsOpen(!isOpen)}
          className={`flex items-center gap-3 px-6 py-3 rounded-2xl font-bold transition-all shadow-xl border ${
            isOpen 
            ? 'bg-brand-gold text-slate-900 border-brand-gold' 
            : 'bg-slate-800 text-brand-gold border-slate-700 hover:border-brand-gold/50'
          }`}
        >
          <Share2 size={20} />
          <span>Share Map</span>
          <ChevronUp size={16} className={`transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} />
        </button>
      </div>
    </div>
  );
};

export default ExportBar;
