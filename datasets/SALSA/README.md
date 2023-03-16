This is the README file for the SALSA dataset http://tev.fbk.eu/salsa

==============
1. VISUAL DATA
==============

Video data for each camera and scenario is provided with a separate .avi file. We used mpeg4 encoding with native resolution 1024x768 at fps=15. All video data is synchronized across cameras. To extract the frames, you can use e.g. ffmpeg tool: ffmpeg -i salsa_cpp_cam1.avi cam1/%06d.jpg

=============
2. BADGE DATA
=============

Each of the badges may have recorded different modalities. If the file corresponding to the modality is not there, it means that the badge did not record correctly that modality. All information is writen in CSV format.

The summary file contains what modality is available in each of the badges.

The first column of all modalities is the timestamp in SECONDS from the first visual images (some badges started working before the video apparently).

Accelerometer fast (accel_fast): body pose in angles.
Accelerometer slow (accel_slow): motion energy, motion energy variation, body pose in angles.
Bluetooth slow (bluetooth_slow): badge ID, signal power.
Audio features fast (feat_fast): min, max, average, variance, starndard deviation, other.
Audio features slow (feat_slow): min, max, average, variance, starndard deviation.
Frequency coefficients fast (freq_fast): 16 x (frequency bin, energy).
Frequency coefficients slow (freq_slow): 16 x (frequency bin, energy).
Infrared hits slow (infrared_slow): badge ID.

===============
3. GROUND TRUTH
===============

3.1 Position/pose
-----------------

Each row of the files in geometryGT has the following format

Timestamp[s]   Ground_Position_X[m]   Ground_Position_Y[m]   Useless_Field   Body_Pose[rad]   Relative_Head2Body_Pose[rad]  Validity[bool]

Therefore, the ground truth head pose is "Body_Pose[rad] + Relative_Head2Body_Pose[rad]"

3.2 F-formation
---------------

Each row of the f-formation file has the following format

Timestamp[s] ID OF THE GROUP MEMBERS SEPARATED BY SPACES

Therefore, the rows vary in length. In addition, there are many rows per timestamps, since there are many groups at every time.
