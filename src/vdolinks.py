YOUTUBE_LIVE_STREAMS = {
    "fresno": "https://www.youtube.com/watch?v=1xl0hX-nF2E",
    "miami": "https://www.youtube.com/watch?v=gCKAn2q35Dw",
}

DEFAULT_YOUTUBE_STREAM_KEY = "fresno"
DEFAULT_YOUTUBE_STREAM = YOUTUBE_LIVE_STREAMS[DEFAULT_YOUTUBE_STREAM_KEY]



# <features>
#     <step order="1">
#         <name>Traffic Light Analysis</name>
#         <description>
#             Analyze the video or scene to determine the locations of traffic lights.
#         </description>
#     </step>

#     <step order="2">
#         <name>Road Structure Detection</name>
#         <description>
#             Detect road crossings, turns, and identify the type of intersection.
#         </description>
#         <intersectionTypes>
#             <type>4-way intersection ( "+" crossroad )</type>
#             <type>3-way intersection ( T or Y-shaped / isopropyl structure )</type>
#         </intersectionTypes>
#     </step>

#     <step order="3">
#         <name>Traffic Light Detector Mapping</name>
#         <description>
#             Draw or map traffic light detection regions based on identified positions 
#             and road structure.
#         </description>
#     </step>

#     <step order="4">
#         <name>Violation Tracking Initialization</name>
#         <description>
#             Start tracking traffic violations based on detected traffic lights 
#             and road rules.
#         </description>
#     </step>
# </features>