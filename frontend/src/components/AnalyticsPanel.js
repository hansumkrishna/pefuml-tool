import React from 'react';
import { useVideoContext } from '../context/VideoContext';
import './AnalyticsPanel.css';

const AnalyticsPanel = () => {
  const { videoData, currentFrame } = useVideoContext();
  
  if (!videoData || currentFrame === null) {
    return null;
  }
  
  // Find analytics data for current frame
  const analytics = videoData.analytics.find(
    data => data.frame === currentFrame
  );
  
  if (!analytics) {
    return (
      <div className="analytics-panel">
        <h3>Pose Analytics</h3>
        <p>No analytics available for this frame</p>
      </div>
    );
  }
  
  return (
    <div className="analytics-panel">
      <h3>Pose Analytics</h3>
      
      <div className="analytics-section">
        <h4>Joint Angles</h4>
        <div className="analytics-data">
          <div className="analytics-item">
            <span>Left Knee Angle:</span>
            <span>{analytics.angles.left_knee.toFixed(2)}°</span>
          </div>
          <div className="analytics-item">
            <span>Right Knee Angle:</span>
            <span>{analytics.angles.right_knee.toFixed(2)}°</span>
          </div>
        </div>
      </div>
      
      <div className="analytics-section">
        <h4>Lunge Analysis</h4>
        <div className="analytics-data">
          <div className="analytics-item">
            <span>Lunge Distance:</span>
            <span>{analytics.lunge_distance.toFixed(2)}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsPanel;