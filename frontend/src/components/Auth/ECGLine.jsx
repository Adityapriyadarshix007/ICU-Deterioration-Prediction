import React from 'react';
import { motion } from 'framer-motion';

const ECGLine = () => (
  <div className="absolute bottom-0 left-0 right-0 h-12 opacity-10">
    <svg viewBox="0 0 1000 50" className="w-full h-full">
      <motion.polyline
        points="0,25 50,25 60,5 70,45 80,25 200,25 250,25 260,10 270,40 280,25 400,25 450,25 460,5 470,45 480,25 600,25 650,25 660,15 670,35 680,25 800,25 850,25 860,5 870,45 880,25 1000,25"
        fill="none"
        stroke="white"
        strokeWidth="2"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 3, ease: "easeInOut" }}
      />
    </svg>
  </div>
);

export default ECGLine;