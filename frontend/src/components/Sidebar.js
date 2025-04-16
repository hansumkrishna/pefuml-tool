import React, { useState } from 'react';
import { useVideoContext } from '../context/VideoContext';
import './Sidebar.css';

const Sidebar = () => {
  const { 
    uploadVideo, 
    videoData, 
    exportMarkedFrames,
    displayOptions, 
    setDisplayOptions,
    metadata, 
    setMetadata,
    markedFrames,
    addMetadataField
  } = useVideoContext();

  const [videoFile, setVideoFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [newMetadataKey, setNewMetadataKey] = useState('');
  const [newMetadataValue, setNewMetadataValue] = useState('');

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setVideoFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!videoFile) return;
    
    setIsLoading(true);
    try {
      await uploadVideo(videoFile);
    } catch (error) {
      console.error("Error uploading video:", error);
      alert("Failed to upload video. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleExport = () => {
    if (markedFrames.length === 0) {
      alert("Please mark at least one frame before exporting");
      return;
    }
    
    exportMarkedFrames();
  };

  const handleAddMetadata = () => {
    if (newMetadataKey.trim() && newMetadataValue.trim()) {
      addMetadataField(newMetadataKey.trim(), newMetadataValue.trim());
      setNewMetadataKey('');
      setNewMetadataValue('');
    }
  };

  return (
    <div className="sidebar">
      <div className="sidebar-section">
        <h3>PEFUML Tool</h3>
        <p>Upload a video to analyze player poses</p>
      </div>
      
      <div className="sidebar-section">
        <h4>Import Video</h4>
        <input 
          type="file" 
          accept="video/*" 
          onChange={handleFileChange} 
          className="file-input"
        />
        <button 
          onClick={handleUpload} 
          disabled={!videoFile || isLoading}
          className="sidebar-button"
        >
          {isLoading ? 'Processing...' : 'Upload & Analyze'}
        </button>
      </div>

      {videoData && (
        <>
          <div className="sidebar-section">
            <h4>Display Options</h4>
            <div className="option-row">
              <input
                type="checkbox"
                id="show-video"
                checked={displayOptions.showVideo}
                onChange={() => setDisplayOptions({
                  ...displayOptions,
                  showVideo: !displayOptions.showVideo
                })}
              />
              <label htmlFor="show-video">Show Video</label>
            </div>
            <div className="option-row">
              <input
                type="checkbox"
                id="show-keypoints"
                checked={displayOptions.showKeypoints}
                onChange={() => setDisplayOptions({
                  ...displayOptions,
                  showKeypoints: !displayOptions.showKeypoints
                })}
              />
              <label htmlFor="show-keypoints">Show 3D Keypoints</label>
            </div>
          </div>

          <div className="sidebar-section">
            <h4>Metadata</h4>
            <div className="metadata-form">
              <input
                type="text"
                placeholder="Key"
                value={newMetadataKey}
                onChange={(e) => setNewMetadataKey(e.target.value)}
              />
              <input
                type="text"
                placeholder="Value"
                value={newMetadataValue}
                onChange={(e) => setNewMetadataValue(e.target.value)}
              />
              <button onClick={handleAddMetadata} className="sidebar-button">Add</button>
            </div>
            
            <div className="metadata-list">
              {Object.entries(metadata).map(([key, value]) => (
                <div key={key} className="metadata-item">
                  <span>{key}: </span>
                  <span>{value}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="sidebar-section">
            <h4>Marked Frames</h4>
            <div className="marked-frames-count">
              Total: {markedFrames.length} frames marked
            </div>
            
            <button 
              onClick={handleExport} 
              disabled={markedFrames.length === 0}
              className="sidebar-button export-button"
            >
              Export Marked Frames
            </button>
          </div>
        </>
      )}
    </div>
  );
};

export default Sidebar;