import React, { useState } from 'react';
import { useVideoContext } from '../context/VideoContext';
import './AnalyticsPanel.css';

const AnalyticsPanel = () => {
  const { videoData, currentFrame } = useVideoContext();
  const [isExpanded, setIsExpanded] = useState(true);

  if (!videoData || currentFrame === null) {
    return null;
  }

  // Find analytics data for current frame
  const analytics = videoData.analytics.find(
    data => data.frame === currentFrame
  );

  // Function to generate readable labels from angle keys
  const formatAngleLabel = (key) => {
    return key
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Format section titles to be more readable
  const formatSectionTitle = (key) => {
    return key
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Format measurement values with proper units
  const formatValue = (key, value) => {
    if (value === undefined || value === null) return 'N/A';

    // Determine appropriate unit based on the type of measurement
    if (key.includes('angle') || key.includes('rotation') ||
        key.includes('flexion') || key.includes('elevation') ||
        key.includes('tilt') || key.includes('azimuth')) {
      return `${value.toFixed(2)}°`;
    } else if (key.includes('distance') || key.includes('height') ||
               key.includes('cog')) {
      return value.toFixed(2); // Units depend on input scale, could be cm/m
    } else if (key.includes('velocity')) {
      return `${value.toFixed(2)} units/s`;
    }

    return value.toFixed(2);
  };

  // Helper to determine if a section should be displayed based on data
  const shouldDisplaySection = (sectionData) => {
    if (!sectionData) return false;
    return Object.values(sectionData).some(value =>
      value !== undefined && value !== null && !isNaN(value)
    );
  };

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="analytics-panel">
      <div className="panel-header">
        <h3>Pose Analytics</h3>
        <button className="toggle-button" onClick={toggleExpand}>
          {isExpanded ? '−' : '+'}
        </button>
      </div>

      {isExpanded && (
        <div className="panel-content">
          {!analytics || !analytics.measurements ? (
            <p>No analytics available for this frame</p>
          ) : (
            <>
              {/* Render each measurement category as a section */}
              {Object.entries(analytics.measurements).map(([sectionKey, sectionData]) => {
                // Skip rendering the section if it's just lunge_distance (we'll handle it separately)
                if (sectionKey === 'lunge_distance') return null;

                // Skip rendering if section has no valid data
                if (!shouldDisplaySection(sectionData)) return null;

                return (
                  <div className="analytics-section" key={sectionKey}>
                    <h4>{formatSectionTitle(sectionKey)}</h4>
                    <div className="analytics-data">
                      {Object.entries(sectionData).map(([key, value]) => {
                        // Skip if value is undefined or null
                        if (value === undefined || value === null || isNaN(value)) return null;

                        return (
                          <div className="analytics-item" key={key}>
                            <span>{formatAngleLabel(key)}:</span>
                            <span>{formatValue(key, value)}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}

              {/* Render lunge distance separately */}
              {analytics.measurements.lunge_distance !== undefined && (
                <div className="analytics-section">
                  <h4>Lunge Analysis</h4>
                  <div className="analytics-data">
                    <div className="analytics-item">
                      <span>Lunge Distance:</span>
                      <span>{analytics.measurements.lunge_distance.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default AnalyticsPanel;