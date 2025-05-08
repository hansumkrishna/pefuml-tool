import React, { useState, useEffect } from 'react';
import { useVideoContext } from '../context/VideoContext';
import './ShotDetection.css';

const ShotDetection = () => {
  const { videoData, currentFrame, markCustomFrames } = useVideoContext();
  const [isExpanded, setIsExpanded] = useState(true);
  const [eventName, setEventName] = useState('');
  const [selectedMetrics, setSelectedMetrics] = useState({});
  const [metricValues, setMetricValues] = useState({});
  const [marginValues, setMarginValues] = useState({});
  const [availableMetrics, setAvailableMetrics] = useState([]);

  // Initialize available metrics when currentFrame changes and has data
  useEffect(() => {
    if (!videoData || !videoData.analytics || currentFrame === null) return;

    const frameAnalytics = videoData.analytics.find(data => data.frame === currentFrame);
    if (!frameAnalytics || !frameAnalytics.measurements) return;

    // Collect all available metrics from all sections
    const metrics = [];

    Object.entries(frameAnalytics.measurements).forEach(([sectionKey, sectionData]) => {
      if (sectionKey === 'lunge_distance') {
        metrics.push({
          id: 'lunge_distance',
          label: 'Lunge Distance',
          value: frameAnalytics.measurements.lunge_distance,
          section: 'Lunge Analysis'
        });
        return;
      }

      Object.entries(sectionData).forEach(([key, value]) => {
        if (value !== undefined && value !== null && !isNaN(value)) {
          metrics.push({
            id: `${sectionKey}.${key}`,
            label: formatMetricLabel(key),
            value: value,
            section: formatSectionTitle(sectionKey)
          });
        }
      });
    });

    setAvailableMetrics(metrics);

    // Initialize selection state
    const initialSelectedState = {};
    const initialValues = {};
    const initialMargins = {};

    metrics.forEach(metric => {
      initialSelectedState[metric.id] = false;
      initialValues[metric.id] = metric.value;
      initialMargins[metric.id] = 10; // Default 10% margin
    });

    setSelectedMetrics(initialSelectedState);
    setMetricValues(initialValues);
    setMarginValues(initialMargins);
  }, [videoData, currentFrame]);

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  // Format section titles to be more readable
  const formatSectionTitle = (key) => {
    return key
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Format metric labels to be more readable
  const formatMetricLabel = (key) => {
    return key
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const handleMetricToggle = (metricId) => {
    setSelectedMetrics(prev => ({
      ...prev,
      [metricId]: !prev[metricId]
    }));
  };

  const handleValueChange = (metricId, value) => {
    setMetricValues(prev => ({
      ...prev,
      [metricId]: parseFloat(value)
    }));
  };

  const handleMarginChange = (metricId, value) => {
    setMarginValues(prev => ({
      ...prev,
      [metricId]: parseFloat(value)
    }));
  };

  // Function to consolidate frames that are within 100 frames of each other
  const consolidateFrames = (frames) => {
    if (!frames || frames.length <= 1) return frames;

    // Sort frames numerically
    const sortedFrames = [...frames].sort((a, b) => a - b);

    // Group frames that are within 100 frames of each other
    const groups = [];
    let currentGroup = [sortedFrames[0]];

    for (let i = 1; i < sortedFrames.length; i++) {
      const currentFrame = sortedFrames[i];
      const lastFrameInGroup = currentGroup[currentGroup.length - 1];

      if (currentFrame - lastFrameInGroup <= 100) {
        // Frame is within range, add to current group
        currentGroup.push(currentFrame);
      } else {
        // Frame is outside range, finish current group and start a new one
        groups.push(currentGroup);
        currentGroup = [currentFrame];
      }
    }

    // Add the last group if it has elements
    if (currentGroup.length > 0) {
      groups.push(currentGroup);
    }

    // Calculate median frame for each group
    const medianFrames = groups.map(group => {
      // Sort the group (should already be sorted but just to be sure)
      group.sort((a, b) => a - b);

      // Find median
      const mid = Math.floor(group.length / 2);
      if (group.length % 2 === 0) {
        // Even number of elements, average the middle two
        return Math.round((group[mid - 1] + group[mid]) / 2);
      } else {
        // Odd number of elements, return the middle one
        return group[mid];
      }
    });

    return medianFrames;
  };

  const handleDetectFrames = () => {
    if (!eventName.trim()) {
      alert("Please enter an event name");
      return;
    }

    // Check if any metrics are selected
    const anySelected = Object.values(selectedMetrics).some(selected => selected);
    if (!anySelected) {
      alert("Please select at least one metric for detection");
      return;
    }

    // Get selected metrics with their values and margins
    const filters = Object.entries(selectedMetrics)
      .filter(([_, selected]) => selected)
      .map(([metricId]) => {
        const value = metricValues[metricId];
        const margin = marginValues[metricId];

        // Calculate min and max allowed values based on margin
        const marginAmount = value * (margin / 100);
        const minValue = value - marginAmount;
        const maxValue = value + marginAmount;

        return {
          id: metricId,
          min: minValue,
          max: maxValue
        };
      });

    // Loop through all frames to find matching frames
    const matchingFrames = [];

    if (videoData && videoData.analytics) {
      videoData.analytics.forEach(frameData => {
        let isMatch = true;

        for (const filter of filters) {
          let actualValue;

          // Extract the actual value from the frame data
          if (filter.id === 'lunge_distance') {
            actualValue = frameData.measurements.lunge_distance;
          } else {
            const [section, metric] = filter.id.split('.');
            if (frameData.measurements[section] &&
                frameData.measurements[section][metric] !== undefined) {
              actualValue = frameData.measurements[section][metric];
            }
          }

          // If we can't find the value or it's outside our range, this frame doesn't match
          if (actualValue === undefined ||
              isNaN(actualValue) ||
              actualValue < filter.min ||
              actualValue > filter.max) {
            isMatch = false;
            break;
          }
        }

        if (isMatch) {
          matchingFrames.push(frameData.frame);
        }
      });
    }

    // Consolidate frames that are within 100 frames of each other
    const consolidatedFrames = consolidateFrames(matchingFrames);

    // Add the matching frames to marked frames
    if (consolidatedFrames.length > 0) {
      markCustomFrames(eventName, consolidatedFrames);

      // Prepare message with original and consolidated counts
      const message = matchingFrames.length === consolidatedFrames.length
        ? `Found ${consolidatedFrames.length} matching frames for "${eventName}"`
        : `Found ${matchingFrames.length} matching frames, consolidated to ${consolidatedFrames.length} for "${eventName}"`;

      alert(message);
    } else {
      alert("No matching frames found. Try adjusting your criteria.");
    }
  };

  // Group metrics by section
  const metricsBySection = availableMetrics.reduce((acc, metric) => {
    if (!acc[metric.section]) {
      acc[metric.section] = [];
    }
    acc[metric.section].push(metric);
    return acc;
  }, {});

  return (
    <div className="shot-detection">
      <div className="panel-header">
        <h3>Shot Detection</h3>
        <button className="toggle-button" onClick={toggleExpand}>
          {isExpanded ? '−' : '+'}
        </button>
      </div>

      {isExpanded && (
        <div className="panel-content">
          <div className="event-name-container">
            <label>Event Name:</label>
            <input
              type="text"
              value={eventName}
              onChange={(e) => setEventName(e.target.value)}
              placeholder="e.g., Perfect Lunge"
              className="event-name-input"
            />
          </div>

          <div className="metrics-container">
            {Object.entries(metricsBySection).map(([section, metrics]) => (
              <div key={section} className="metrics-section">
                <h4>{section}</h4>
                {metrics.map(metric => (
                  <div key={metric.id} className="metric-item">
                    <div className="metric-header">
                      <input
                        type="checkbox"
                        checked={selectedMetrics[metric.id] || false}
                        onChange={() => handleMetricToggle(metric.id)}
                      />
                      <span className="metric-label">{metric.label}</span>
                    </div>
                    <div className="metric-controls">
                      <div className="value-control">
                        <label>Value:</label>
                        <input
                          type="number"
                          value={metricValues[metric.id] || 0}
                          onChange={(e) => handleValueChange(metric.id, e.target.value)}
                          disabled={!selectedMetrics[metric.id]}
                          step="0.01"
                        />
                      </div>
                      <div className="margin-control">
                        <label>Margin (%):</label>
                        <input
                          type="number"
                          value={marginValues[metric.id] || 10}
                          onChange={(e) => handleMarginChange(metric.id, e.target.value)}
                          disabled={!selectedMetrics[metric.id]}
                          min="0"
                          max="100"
                          step="1"
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </div>

          <button
            className="detect-button"
            onClick={handleDetectFrames}
          >
            Find Matching Frames
          </button>
        </div>
      )}
    </div>
  );
};

export default ShotDetection;