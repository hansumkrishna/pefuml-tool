import React from 'react';
import { useVideoContext } from '../context/VideoContext';
import './FrameMarker.css';

const FrameMarker = () => {
  const { 
    videoData, 
    markedFrames, 
    currentFrame, 
    setCurrentFrame, 
    removeMarkedFrame 
  } = useVideoContext();
  
  if (!videoData || markedFrames.length === 0) {
    return (
      <div className="frame-marker">
        <h3>Marked Frames</h3>
        <p>No frames marked yet</p>
        <p>Use the "Mark Frame" button to save important positions</p>
      </div>
    );
  }
  
  const jumpToFrame = (frame) => {
    setCurrentFrame(frame);
  };
  
  const handleRemoveFrame = (e, frameId) => {
    e.stopPropagation();
    if (window.confirm("Are you sure you want to remove this marked frame?")) {
      removeMarkedFrame(frameId);
    }
  };
  
  return (
    <div className="frame-marker">
      <h3>Marked Frames</h3>
      <div className="marked-frames-list">
        {markedFrames.map((mark) => (
          <div 
            key={mark.id}
            className={`marked-frame-item ${mark.frame === currentFrame ? 'active' : ''}`}
            onClick={() => jumpToFrame(mark.frame)}
          >
            <div className="marked-frame-info">
              <span className="marked-frame-name">{mark.name}</span>
              <span className="marked-frame-number">Frame: {mark.frame}</span>
            </div>
            <button 
              className="remove-frame-button"
              onClick={(e) => handleRemoveFrame(e, mark.id)}
            >
              &times;
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FrameMarker;