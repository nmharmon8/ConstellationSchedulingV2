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
                        r={5}
                        className="downlink-base"
                        style={{ 
                            stroke: color,
                            strokeWidth: "0.1",
                            fill: "rgba(255, 255, 255, 0.3)" 
                        }}
                    />
                    
                    {/* Base Support Structure */}
                    {/* <rect
                        x="-1.5"
                        y="-4"
                        width="3"
                        height="4"
                        style={{
                            fill: color,
                            opacity: "0.8"
                        }}
                    /> */}
                    
                    {/* Satellite Dish */}
                    <g className="radar-dish">
                        <g transform="translate(0, -4) rotate(0)">
                            {/* Main Dish Parabola */}
                            <path
                                d="M 10 0 Q 0 12 -10 0"
                                className="dish-main"
                                style={{ 
                                    stroke: color,
                                    strokeWidth: "1.5",
                                    fill: "rgba(255, 255, 255, 0.2)"
                                }}
                            />
                            {/* Dish Feed Horn */}
                            <circle
                                cx="0"
                                cy="4"
                                r="1.5"
                                style={{
                                    fill: color,
                                    opacity: "0.9"
                                }}
                            />
                            {/* Support Struts */}
                            <line 
                                x1="-6" 
                                y1="1" 
                                x2="0" 
                                y2="4" 
                                className="dish-support"
                                style={{ 
                                    stroke: color,
                                    strokeWidth: "1"
                                }}
                            />
                            <line 
                                x1="6" 
                                y1="1" 
                                x2="0" 
                                y2="4" 
                                className="dish-support"
                                style={{ 
                                    stroke: color,
                                    strokeWidth: "1"
                                }}
                            />
                        </g>
                    </g>
                </g>
            </g>
        </Marker>
    );
});

export default DownlinkStation; 