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
                </g>
            </g>
        </Marker>
    );
});

export default DownlinkStation; 