import React, { createContext, useContext, useState } from 'react';
import { uploadVideoService, exportFramesService } from '../services/api';

const VideoContext = createContext();

export const useVideoContext = () => useContext(VideoContext);

export const VideoProvider = ({ children }) => {
  const [videoData, setVideoData] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [markedFrames, setMarkedFrames] = useState([]);
  const [metadata, setMetadata] = useState({});
  const [displayOptions, setDisplayOptions] = useState({
    showVideo: true,
    showKeypoints: true
  });

  // Upload and process video
  const uploadVideo = async (videoFile) => {
    const data = await uploadVideoService(videoFile);
    setVideoData(data);
    setCurrentFrame(0);
    setMarkedFrames([]);
    setMetadata({});
    return data;
  };

  // Mark current frame
  const markCurrentFrame = (name) => {
    if (currentFrame === null) return;

    const frameId = Date.now().toString();
    const newMarkedFrame = {
      id: frameId,
      name: name,
      frame: currentFrame,
      metadata: {}
    };

    setMarkedFrames([...markedFrames, newMarkedFrame]);
  };

  // Mark multiple frames (for shot detection)
  const markCustomFrames = (name, framesList) => {
    if (!framesList || framesList.length === 0) return;

    const newMarkedFrames = framesList.map(frame => ({
      id: Date.now().toString() + '_' + frame, // Ensure unique IDs
      name: `${name} (Frame ${frame})`,
      frame: frame,
      metadata: {} // Maintain structure consistency
    }));

    setMarkedFrames(prev => [...prev, ...newMarkedFrames]);
  };

  // Remove marked frame
  const removeMarkedFrame = (frameId) => {
    setMarkedFrames(markedFrames.filter(frame => frame.id !== frameId));
  };

  // Add metadata field
  const addMetadataField = (key, value) => {
    setMetadata({
      ...metadata,
      [key]: value
    });
  };

  // Export marked frames
  const exportMarkedFrames = async () => {
    if (!videoData || markedFrames.length === 0) return;

    try {
      const exportData = {
        video_id: videoData.id,
        marked_frames: markedFrames,
        metadata: metadata
      };

      const response = await exportFramesService(exportData);

      // Create download link for the exported JSON
      const jsonString = JSON.stringify(response.export_data, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);

      const a = document.createElement('a');
      a.href = url;
      a.download = `pefuml_export_${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();

      URL.revokeObjectURL(url);
      document.body.removeChild(a);

      alert("Export completed successfully!");
    } catch (error) {
      console.error("Error exporting frames:", error);
      alert("Failed to export marked frames. Please try again.");
    }
  };

  return (
    <VideoContext.Provider value={{
      videoData,
      currentFrame,
      setCurrentFrame,
      isPlaying,
      setIsPlaying,
      markedFrames,
      markCurrentFrame,
      markCustomFrames, // Added the new function
      removeMarkedFrame,
      metadata,
      setMetadata,
      addMetadataField,
      displayOptions,
      setDisplayOptions,
      uploadVideo,
      exportMarkedFrames
    }}>
      {children}
    </VideoContext.Provider>
  );
};

export default VideoProvider;