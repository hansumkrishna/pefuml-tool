const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Upload and process video
export const uploadVideoService = async (videoFile) => {
  const formData = new FormData();
  formData.append('video', videoFile);
  
  try {
    const response = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to upload video');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error uploading video:', error);
    throw error;
  }
};

// Get video data by ID
export const getVideoDataService = async (videoId) => {
  try {
    const response = await fetch(`${API_URL}/videos/${videoId}`);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to get video data');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error getting video data:', error);
    throw error;
  }
};

// Export marked frames
export const exportFramesService = async (exportData) => {
  try {
    const response = await fetch(`${API_URL}/export`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(exportData)
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to export frames');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error exporting frames:', error);
    throw error;
  }
};