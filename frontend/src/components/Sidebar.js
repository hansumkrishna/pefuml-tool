import React, { useState, useEffect } from 'react';
import { useVideoContext } from '../context/VideoContext';
import { getVideosService } from '../services/api';
import './Sidebar.css';

const Sidebar = () => {
  const {
    uploadVideo,
    videoData,
    exportMarkedFrames,
    displayOptions,
    setDisplayOptions,
    metadata,
    markedFrames,
    addMetadataField,
    loadVideoById
  } = useVideoContext();

  const [videoFile, setVideoFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [newMetadataKey, setNewMetadataKey] = useState('');
  const [newMetadataValue, setNewMetadataValue] = useState('');

  // New states for previous videos feature
  const [previousVideos, setPreviousVideos] = useState([]);
  const [isLoadingPrevious, setIsLoadingPrevious] = useState(false);
  // Add new state for search query
  const [searchQuery, setSearchQuery] = useState('');

  // Fetch the list of previous videos on component mount
  useEffect(() => {
    fetchPreviousVideos();
  }, []);

  // Filter previous videos based on search query
  const filteredVideos = previousVideos.filter(video =>
    video.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
    video.id.toString().includes(searchQuery)
  );

  // Replace your existing fetchPreviousVideos function with:
  const fetchPreviousVideos = async () => {
    setIsLoadingPrevious(true);
    try {
      const videos = await getVideosService();
      setPreviousVideos(videos);
    } catch (error) {
      console.error('Error fetching previous videos:', error);
    } finally {
      setIsLoadingPrevious(false);
    }
  };

  const handleLoadPreviousVideo = async (videoId) => {
    setIsLoading(true);
    try {
      await loadVideoById(videoId);
    } catch (error) {
      console.error("Error loading previous video:", error);
      alert("Failed to load video. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

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
      // Refresh the list of previous videos after successful upload
      fetchPreviousVideos();
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

  // Format the date for display
  const formatDate = (timestamp) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
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

      {/* Enhanced section for previous videos */}
      <div className="sidebar-section previous-videos-section">
        <h4>Previous Videos</h4>

        {/* Add search input */}
        <div className="search-container">
          <input
            type="text"
            placeholder="Search by name or ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
          {searchQuery && (
            <button
              className="clear-search-button"
              onClick={() => setSearchQuery('')}
            >
              ✕
            </button>
          )}
        </div>

        {isLoadingPrevious ? (
          <p className="loading-message">Loading previous videos...</p>
        ) : previousVideos.length > 0 ? (
          <>
            <div className="previous-videos-list">
              {filteredVideos.map(video => (
                <div key={video.id} className="previous-video-item">
                  <div className="video-info">
                    <div className="video-name">{video.filename}</div>
                    <div className="video-details">
                      <span className="detail-item">
                        <span className="detail-icon">🎞️</span>
                        {video.frame_count} frames
                      </span>
                      <span className="detail-item">
                        <span className="detail-icon">⏱️</span>
                        {video.fps} FPS
                      </span>
                    </div>
                    <div className="video-date">
                      <span className="date-icon">📅</span>
                      {formatDate(video.processed_date)}
                    </div>
                  </div>
                  <button
                    onClick={() => handleLoadPreviousVideo(video.id)}
                    disabled={isLoading}
                    className="load-video-button"
                  >
                    Load
                  </button>
                </div>
              ))}
            </div>
            {/* Show filtered count message */}
            {filteredVideos.length === 0 ? (
              <p className="no-results-message">No videos match your search</p>
            ) : (
              <div className="video-count">
                Showing {filteredVideos.length} of {previousVideos.length} videos
              </div>
            )}
          </>
        ) : (
          <p className="no-videos-message">No previously processed videos found</p>
        )}
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