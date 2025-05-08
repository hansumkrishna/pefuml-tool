import React, { useState } from 'react';
import { useVideoContext } from '../context/VideoContext';
import './FrameMarker.css';

const FrameMarker = () => {
  const {
    videoData,
    markedFrames,
    currentFrame,
    setCurrentFrame,
    removeMarkedFrame,
    clearAllMarkedFrames
  } = useVideoContext();
  const [isExpanded, setIsExpanded] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  const jumpToFrame = (frame) => {
    setCurrentFrame(frame);
  };

  const handleRemoveFrame = (e, frameId) => {
    e.stopPropagation();
    removeMarkedFrame(frameId);
  };

  const handleRemoveAllFrames = () => {
    if (markedFrames.length === 0) return;

    if (window.confirm("Are you sure you want to remove all marked frames? This action cannot be undone.")) {
      clearAllMarkedFrames();
    }
  };

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  const filteredFrames = markedFrames.filter((mark) =>
    mark.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="frame-marker">
      <div className="panel-header">
        <h3>Marked Frames</h3>
        <button className="toggle-button" onClick={toggleExpand}>
          {isExpanded ? '−' : '+'}
        </button>
      </div>

      {isExpanded && (
        <div className="panel-content">
          {!videoData || markedFrames.length === 0 ? (
            <div>
              <p>No frames marked yet</p>
              <p>Use the "Mark Frame" button to save important positions</p>
            </div>
          ) : (
            <>
              <div className="search-container">
                <input
                  type="text"
                  placeholder="Search frames..."
                  value={searchTerm}
                  onChange={handleSearchChange}
                  className="search-input"
                />
              </div>
              <div className="marked-frames-header">
                <span>{filteredFrames.length} of {markedFrames.length} marked frame(s)</span>
                <span></span>
                <button
                  className="remove-all-button"
                  onClick={handleRemoveAllFrames}
                >
                  Remove All
                </button>
              </div>
              <div className="marked-frames-list">
                {filteredFrames.map((mark) => (
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
              {searchTerm && filteredFrames.length === 0 && (
                <div className="no-results">
                  <p>No frames match your search</p>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default FrameMarker;