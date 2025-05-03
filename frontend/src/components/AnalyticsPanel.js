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
  
  // Function to generate readable labels from angle keys
  const formatAngleLabel = (key) => {
    return key
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Group angles by body part for better organization
  const groupAngles = () => {
    const groups = {
      'Torso & Neck': [],
      'Arms & Shoulders': [],
      'Legs & Hips': [],
      'Other': []
    };

    if (!analytics.angles) return groups;

    Object.entries(analytics.angles).forEach(([key, value]) => {
      // Skip if value is undefined or null
      if (value === undefined || value === null) return;

      const formattedLabel = formatAngleLabel(key);
      const angleItem = {
        key,
        label: formattedLabel,
        value: typeof value === 'number' ? value.toFixed(2) : value
      };

      // Categorize angles into groups
      if (key.includes('torso') || key.includes('neck')) {
        groups['Torso & Neck'].push(angleItem);
      } else if (key.includes('shoulder') || key.includes('elbow') || key.includes('arm')) {
        groups['Arms & Shoulders'].push(angleItem);
      } else if (key.includes('knee') || key.includes('hip') || key.includes('ankle')) {
        groups['Legs & Hips'].push(angleItem);
      } else {
        groups['Other'].push(angleItem);
      }
    });

    return groups;
  };

  const angleGroups = groupAngles();

  return (
    <div className="analytics-panel">
      <h3>Pose Analytics</h3>

      {/* Render all angle groups */}
      {Object.entries(angleGroups).map(([groupName, angles]) => {
        if (angles.length === 0) return null;

        return (
          <div className="analytics-section" key={groupName}>
            <h4>{groupName}</h4>
            <div className="analytics-data">
              {angles.map(angle => (
                <div className="analytics-item" key={angle.key}>
                  <span>{angle.label}:</span>
                  <span>{angle.value}°</span>
                </div>
              ))}
            </div>
          </div>
        );
      })}

      {/* Keep existing lunge analysis section */}
      {analytics.lunge_distance !== undefined && (
        <div className="analytics-section">
          <h4>Lunge Analysis</h4>
          <div className="analytics-data">
            <div className="analytics-item">
              <span>Lunge Distance:</span>
              <span>{analytics.lunge_distance.toFixed(2)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalyticsPanel;