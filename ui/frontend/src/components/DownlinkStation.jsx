import React from 'react';
import { Marker } from "react-simple-maps";
import './DownlinkStation.css';

const DownlinkStation = React.memo(({ color, task }) => {
    // Extract coordinates and tooltip content from task
    const coordinates = [task.longitude, task.latitude];
    const tooltipContent = `${task.task_type_str} (Priority: ${task.priority.toFixed(2)})`;

    return (
        <Marker
            coordinates={coordinates}
            data-tooltip-id="task-tooltip"
            data-tooltip-content={tooltipContent}
        >
            <g>
                <g className="downlink-station">
                    {/* Ground Base Platform */}
                    <circle
                        r={4}
                        className="downlink-base"
                        style={{ stroke: color }}
                    />
                    
                    {/* Satellite Dish */}
                    <g className="radar-dish">
                        <g transform="translate(0, -4)">
                            <path
                                d="M -4 0 Q 0 -6 4 0"
                                className="dish-main"
                                style={{ stroke: color }}
                            />
                            <line 
                                x1="0" 
                                y1="0" 
                                x2="0" 
                                y2="2" 
                                className="dish-support"
                                style={{ stroke: color }}
                            />
                        </g>
                    </g>
                </g>
            </g>
        </Marker>
    );
});

export default DownlinkStation; 