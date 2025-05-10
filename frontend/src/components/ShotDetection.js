import React, { useState, useEffect, useRef } from 'react';
import { useVideoContext } from '../context/VideoContext';
import './ShotDetection.css';

const ShotDetection = () => {
  const { videoData, currentFrame, markCustomFrames } = useVideoContext();
  const [isExpanded, setIsExpanded] = useState(true);
  const [eventName, setEventName] = useState('');
  const [selectedMetrics, setSelectedMetrics] = useState({});
  const [metricValues, setMetricValues] = useState({});
  const [globalMargin, setGlobalMargin] = useState(10); // Default 10% margin
  const [frameWindowSize, setFrameWindowSize] = useState(30); // Default window size of 30 frames
  const [availableMetrics, setAvailableMetrics] = useState([]);
  const [searchTerm, setSearchTerm] = useState(''); // New state for search functionality

  // Reference to track if initial values have been set
  const initialValuesSet = useRef(false);

  // Filter metrics based on search term
  const filteredMetrics = availableMetrics.filter(metric =>
    metric.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
    metric.section.toLowerCase().includes(searchTerm.toLowerCase())
  );

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

    // Initialize metrics only if they haven't been set yet
    if (!initialValuesSet.current) {
      // Initialize selection state and values with min/max range
      const initialSelectedState = {};
      const initialValues = {};

      metrics.forEach(metric => {
        initialSelectedState[metric.id] = false;
        const currentValue = metric.value;
        const marginAmount = currentValue * (globalMargin / 100);

        initialValues[metric.id] = {
          current: currentValue,
          min: currentValue - marginAmount,
          max: currentValue + marginAmount
        };
      });

      setSelectedMetrics(initialSelectedState);
      setMetricValues(initialValues);
      initialValuesSet.current = true;
    } else {
      // Update only the values for metrics that are not currently selected
      // or for newly available metrics not in our state yet
      setMetricValues(prev => {
        const updated = { ...prev };

        metrics.forEach(metric => {
          // If this is a new metric or the metric is disabled, set/update its value
          if (!updated[metric.id] || !selectedMetrics[metric.id]) {
            const currentValue = metric.value;
            const marginAmount = currentValue * (globalMargin / 100);

            updated[metric.id] = {
              current: currentValue,
              min: currentValue - marginAmount,
              max: currentValue + marginAmount
            };
          }
        });

        return updated;
      });

      // Update selectedMetrics for any new metrics
      setSelectedMetrics(prev => {
        const updated = { ...prev };

        metrics.forEach(metric => {
          if (updated[metric.id] === undefined) {
            updated[metric.id] = false;
          }
        });

        return updated;
      });
    }
  }, [videoData, currentFrame, globalMargin, selectedMetrics]);

  // Update min/max values when global margin changes
  useEffect(() => {
    if (availableMetrics.length === 0) return;

    setMetricValues(prev => {
      const updated = { ...prev };

      // Only update min/max for metrics with auto-margin (disabled metrics or ones we want to auto-update)
      Object.keys(updated).forEach(metricId => {
        // Only update disabled metrics or ones that don't have custom values
        if (!selectedMetrics[metricId]) {
          const currentValue = updated[metricId].current;
          const marginAmount = currentValue * (globalMargin / 100);

          updated[metricId] = {
            ...updated[metricId],
            min: currentValue - marginAmount,
            max: currentValue + marginAmount
          };
        }
      });

      return updated;
    });
  }, [globalMargin, availableMetrics, selectedMetrics]);

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

  const handleValueChange = (metricId, field, value) => {
    setMetricValues(prev => {
      const updated = { ...prev };
      updated[metricId] = { ...updated[metricId], [field]: parseFloat(value) };

      // If we're updating the current value, also update min/max based on margin
      if (field === 'current') {
        const marginAmount = parseFloat(value) * (globalMargin / 100);
        updated[metricId].min = parseFloat(value) - marginAmount;
        updated[metricId].max = parseFloat(value) + marginAmount;
      }

      return updated;
    });
  };

  const handleGlobalMarginChange = (value) => {
    setGlobalMargin(parseFloat(value));
  };

  const handleFrameWindowSizeChange = (value) => {
    setFrameWindowSize(parseInt(value, 10));
  };

  // Function to check if all selected metrics appear anywhere within a window of frames
  const checkMetricsInFrameWindow = (startFrame, endFrame, filters) => {
    if (!videoData || !videoData.analytics) return false;

    // Get all the frames in this window
    const framesInWindow = videoData.analytics.filter(
      frameData => frameData.frame >= startFrame && frameData.frame <= endFrame
    );

    if (framesInWindow.length === 0) return false;

    // Check if each filter is satisfied by at least one frame in the window
    const satisfiedFilters = filters.map(filter => {
      return framesInWindow.some(frameData => {
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

        // Check if value is within the min-max range
        return actualValue !== undefined &&
               !isNaN(actualValue) &&
               actualValue >= filter.min &&
               actualValue <= filter.max;
      });
    });

    // All filters must be satisfied by at least one frame in the window
    return satisfiedFilters.every(isSatisfied => isSatisfied);
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

    // Validate frame window size
    if (frameWindowSize <= 0) {
      alert("Frame window size must be a positive number");
      return;
    }

    // Get selected metrics with their min/max values
    const filters = Object.entries(selectedMetrics)
      .filter(([_, selected]) => selected)
      .map(([metricId]) => {
        return {
          id: metricId,
          min: metricValues[metricId].min,
          max: metricValues[metricId].max
        };
      });

    // Find matching frame windows
    const matchingWindowStarts = [];

    if (videoData && videoData.analytics) {
      // Get all frame numbers and sort them
      const allFrames = videoData.analytics.map(data => data.frame).sort((a, b) => a - b);

      // Check each possible window start position
      for (let i = 0; i < allFrames.length; i++) {
        const startFrame = allFrames[i];
        const endFrame = startFrame + frameWindowSize - 1;

        // Skip if the end frame would be beyond the last available frame
        if (endFrame > allFrames[allFrames.length - 1]) break;

        // Check if all selected metrics appear somewhere in this frame window
        if (checkMetricsInFrameWindow(startFrame, endFrame, filters)) {
          matchingWindowStarts.push(startFrame);
        }
      }
    }

    // Consolidate frame windows that are close to each other
    const consolidatedFrames = consolidateFrames(matchingWindowStarts);

    // Add the matching frames to marked frames
    if (consolidatedFrames.length > 0) {
      markCustomFrames(eventName, consolidatedFrames);

      // Prepare message with original and consolidated counts
      const message = matchingWindowStarts.length === consolidatedFrames.length
        ? `Found ${consolidatedFrames.length} matching frame sequences for "${eventName}"`
        : `Found ${matchingWindowStarts.length} matching frame sequences, consolidated to ${consolidatedFrames.length} for "${eventName}"`;

      alert(message);
    } else {
      alert("No matching frame sequences found. Try adjusting your criteria.");
    }
  };

  // NEW FUNCTION: Export filter settings to JSON
  const exportFilter = () => {
    // Create the export object following our defined schema
    const exportData = {
      version: "1.0",
      name: eventName || "Unnamed Filter",
      globalMargin,
      frameWindowSize,
      metrics: {}
    };

    // Add all metric configurations
    Object.keys(selectedMetrics).forEach(metricId => {
      if (metricValues[metricId]) {
        exportData.metrics[metricId] = {
          selected: selectedMetrics[metricId],
          min: metricValues[metricId].min,
          current: metricValues[metricId].current,
          max: metricValues[metricId].max
        };
      }
    });

    // Convert to JSON and create download link
    const jsonString = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    // Create a temporary link and trigger download
    const a = document.createElement('a');
    a.href = url;
    a.download = `${exportData.name.replace(/\s+/g, '_')}_filter.json`;
    document.body.appendChild(a);
    a.click();

    // Clean up
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // NEW FUNCTION: Import filter settings from JSON
  const handleFileImport = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importData = JSON.parse(e.target.result);

        // Validate the import data has the expected format
        if (!importData.version || !importData.metrics) {
          throw new Error("Invalid filter file format");
        }

        // Apply the imported settings
        if (importData.name) setEventName(importData.name);
        if (importData.globalMargin) setGlobalMargin(importData.globalMargin);
        if (importData.frameWindowSize) setFrameWindowSize(importData.frameWindowSize);

        // Apply metric selections and values
        const newSelectedMetrics = { ...selectedMetrics };
        const newMetricValues = { ...metricValues };

        Object.entries(importData.metrics).forEach(([metricId, settings]) => {
          // Only apply if the metric exists in our current context
          if (newMetricValues[metricId] !== undefined) {
            newSelectedMetrics[metricId] = settings.selected;
            newMetricValues[metricId] = {
              min: settings.min,
              current: settings.current,
              max: settings.max
            };
          }
        });

        setSelectedMetrics(newSelectedMetrics);
        setMetricValues(newMetricValues);

        alert("Filter settings imported successfully");
      } catch (error) {
        alert(`Error importing filter settings: ${error.message}`);
        console.error("Import error:", error);
      }
    };

    reader.readAsText(file);

    // Reset the file input so the same file can be selected again
    event.target.value = null;
  };

  // Hidden file input for import
  const fileInputRef = useRef(null);

  // Trigger file input click
  const triggerImportDialog = () => {
    fileInputRef.current.click();
  };

  // Group metrics by section
  const metricsBySection = filteredMetrics.reduce((acc, metric) => {
    if (!acc[metric.section]) {
      acc[metric.section] = [];
    }
    acc[metric.section].push(metric);
    return acc;
  }, {});

  return (
    <div className="frame-marker shot-detection">
      <div className="panel-header">
        <h3>Shot Detection</h3>
        <button className="toggle-button" onClick={toggleExpand}>
          {isExpanded ? '−' : '+'}
        </button>
      </div>

      {isExpanded && (
        <div className="panel-content">
          {!videoData ? (
            <div>
              <p>No video data loaded</p>
              <p>Load a video to use shot detection features</p>
            </div>
          ) : (
            <>
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

              <div className="global-controls">
                <div className="frame-window-control">
                  <label>Frame Window Size:</label>
                  <input
                    type="text"
                    value={frameWindowSize}
                    onChange={(e) => handleFrameWindowSizeChange(e.target.value)}
                    min="1"
                    className="frame-window-input"
                  />
                  <span className="input-info">frames</span>
                </div>
                <div className="global-margin-control">
                  <label>Global Margin:</label>
                  <input
                    type="text"
                    value={globalMargin}
                    onChange={(e) => handleGlobalMarginChange(e.target.value)}
                    min="0"
                    max="1000"
                    className="global-margin-input"
                  />
                  <span className="input-info">%</span>
                </div>
              </div>

              {/* NEW: Import/Export buttons */}
              <div className="import-export-controls">
                <button
                  className="import-button"
                  onClick={triggerImportDialog}
                >
                  Import Filter
                </button>
                <button
                  className="export-button"
                  onClick={exportFilter}
                >
                  Export Filter
                </button>
                {/* Hidden file input */}
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileImport}
                  accept=".json"
                  style={{ display: 'none' }}
                />
              </div>

              <div className="detection-info">
                <p className="detection-explanation">
                  Detection will find frame sequences of {frameWindowSize} frames where all selected metrics
                  appear within their min/max ranges (in any order within the sequence).
                </p>
              </div>

              <div className="metrics-header">
                <span>{Object.values(selectedMetrics).filter(selected => selected).length} metric(s) selected</span>
                <button
                  className="detect-button"
                  onClick={handleDetectFrames}
                >
                  Find Matching Sequences
                </button>
              </div>

              {/* NEW: Search field */}
              <div className="search-container">
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search metrics..."
                  className="search-input"
                />
              </div>

              <div className="metrics-list">
                {Object.entries(metricsBySection).map(([section, metrics]) => (
                  <div key={section} className="metrics-section">
                    <h4 className="section-title">{section}</h4>
                    {metrics.map(metric => (
                      <div key={metric.id} className="metric-item">
                        <div className="metric-header">
                          <input
                            type="checkbox"
                            checked={selectedMetrics[metric.id] || false}
                            onChange={() => handleMetricToggle(metric.id)}
                            id={`metric-${metric.id}`}
                          />
                          <label
                            htmlFor={`metric-${metric.id}`}
                            className="metric-label"
                          >
                            {metric.label}
                          </label>
                        </div>
                        <div className="metric-controls">
                          <div className="range-values">
                            <div className="min-value">
                              <label>Min:</label>
                              <input
                                type="text"
                                value={metricValues[metric.id]?.min.toFixed(2) || 0}
                                onChange={(e) => handleValueChange(metric.id, 'min', e.target.value)}
                                disabled={!selectedMetrics[metric.id]}
                              />
                            </div>
                            <div className="current-value">
                              <label>Current:</label>
                              <input
                                type="text"
                                value={metricValues[metric.id]?.current.toFixed(2) || 0}
                                onChange={(e) => handleValueChange(metric.id, 'current', e.target.value)}
                                disabled={!selectedMetrics[metric.id]}
                              />
                            </div>
                            <div className="max-value">
                              <label>Max:</label>
                              <input
                                type="text"
                                value={metricValues[metric.id]?.max.toFixed(2) || 0}
                                onChange={(e) => handleValueChange(metric.id, 'max', e.target.value)}
                                disabled={!selectedMetrics[metric.id]}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default ShotDetection;