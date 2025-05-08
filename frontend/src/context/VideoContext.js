import React, { createContext, useContext, useState } from 'react';
import { uploadVideoService, exportFramesService, getVideoDataService } from '../services/api';

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

// Generate a unique ID for marked frames
const generateUniqueId = () => {
  // Using a combination of timestamp and random number for uniqueness
  return `frame_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};


// Add a marked frame with specified frame number and name
const addMarkedFrame = (frameNumber, frameName) => {
  // Create a new marked frame object
  const newFrame = {
    id: generateUniqueId(),
    frame: frameNumber,
    name: frameName || `Frame ${frameNumber}`  // Default name if none provided
  };

  // Add to the markedFrames array
  setMarkedFrames(prevFrames => [...prevFrames, newFrame]);

  return newFrame; // Return the newly created frame
};


  // Remove marked frame
  const removeMarkedFrame = (frameId) => {
    setMarkedFrames(markedFrames.filter(frame => frame.id !== frameId));
  };

  // Clear all marked frames
  const clearAllMarkedFrames = () => {
    setMarkedFrames([]);
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

  // Import marked frames from JSON
const importMarkedFrames = async (jsonData) => {
  if (!videoData) {
    throw new Error("No video is currently loaded");
  }

  try {
    // Validate the JSON structure
    if (!jsonData || !jsonData.video_id || !Array.isArray(jsonData.marked_frames)) {
      throw new Error("Invalid JSON format. The file is missing required fields.");
    }

    // Check if the video ID matches the currently loaded video
    if (jsonData.video_id !== videoData.id) {
      // Present a confirmation dialog since the video IDs don't match
      const confirmImport = window.confirm(
        "The frames in this file are from a different video. Importing may result in frames pointing to incorrect positions. Do you want to continue?"
      );

      if (!confirmImport) {
        throw new Error("Import cancelled by user");
      }
    }

    // Process and import the marked frames
    // First, clear existing frames if the user confirms
    if (markedFrames.length > 0) {
      const confirmReplace = window.confirm(
        `You currently have ${markedFrames.length} marked frames. Do you want to replace them with the ${jsonData.marked_frames.length} imported frames? Click Cancel to merge instead.`
      );

      if (confirmReplace) {
        // Clear all existing frames
        clearAllMarkedFrames();
      }
    }

    // Import the frames
    // We need to ensure each frame has a unique ID, so we'll generate new IDs for imported frames
    // while preserving the frame number and name
    const importedFrames = jsonData.marked_frames.map(frame => ({
      id: generateUniqueId(), // This function should exist in your context to create unique IDs
      frame: frame.frame,     // The actual frame number
      name: frame.name        // The name of the marker
    }));

    // Add all the imported frames to the current state
    importedFrames.forEach(frame => {
      // Check if frame already exists to avoid duplicates
      const exists = markedFrames.some(existing =>
        existing.frame === frame.frame && existing.name === frame.name
      );

      if (!exists) {
        addMarkedFrame(frame.frame, frame.name);
      }
    });

    // Import metadata if available
    if (jsonData.metadata && typeof jsonData.metadata === 'object') {
      Object.entries(jsonData.metadata).forEach(([key, value]) => {
        // Only add metadata that doesn't already exist
        if (!(key in metadata)) {
          addMetadataField(key, value);
        }
      });
    }

    return {
      status: 'success',
      message: `Successfully imported ${importedFrames.length} frames`,
      importedFrames: importedFrames
    };
  } catch (error) {
    console.error("Error importing frames:", error);
    throw new Error(`Failed to import marked frames: ${error.message}`);
  }
};

    // Update loadVideoById to use getVideoDataService
    const loadVideoById = async (videoId) => {
      try {
        // Use the service function instead of direct fetch
        const data = await getVideoDataService(videoId);

        // Set the video data in context
        setVideoData({
          id: videoId,
          ...data
        });

        // Reset marked frames
        setMarkedFrames([]);

        // Reset metadata or initialize with any existing metadata from the video
        setMetadata(data.metadata || {});

        return data;
      } catch (error) {
        console.error('Error loading video by ID:', error);
        throw error;
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
      markCustomFrames,
      removeMarkedFrame,
      loadVideoById,
      clearAllMarkedFrames,
      metadata,
      setMetadata,
      addMetadataField,
      displayOptions,
      setDisplayOptions,
      uploadVideo,
      exportMarkedFrames,
      importMarkedFrames
    }}>
      {children}
    </VideoContext.Provider>
  );
};

export default VideoProvider;