# Import required modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as ani
import os
import plot_colors as pc
import datetime
import scipy.interpolate

# Define directories
inpath_delta = '/media/reagan/Data/BTSync/Research/Data/GeoNet/delta'
recordpath = '../Records'
modelpath = '../Models'

# Define parameters
hpad = 10.0
vpad = 5.0
scale = {
    'AVAB_1': 500.0,
    'AVAB_2': 500.0,
    'CPLB_1': 50.0,
    'CPLB_2': 50.0,
    'CPXB'  : 50.0,
    'MJCB'  : 100.0,
    'NMIB'  : 500.0,
    'STSB'  : 500.0,
    'VUWB'  : 100.0,
    'WHSB'  : 500.0
}

# Define the animation parameters
bitrate = 3000
fps = 30.0
dt_int = 1.0/fps

# Define the date-time converter
date_time_conv = lambda x: np.array(x[:-1], dtype='datetime64[s]')

# Loop over all models
#modeldirs = sorted([x for x in os.listdir(recordpath)])
modeldirs = ['CPXB']
for modeldir in modeldirs:
    print modeldir

    # Check if the raw accelerograms have been processed
    indir_proc = os.path.join(recordpath, modeldir, 'Processed_Dis')
    indir_level_acc = os.path.join(recordpath, modeldir, 'Level_AbsAcc')
    indir_level_dis = os.path.join(recordpath, modeldir, 'Level_AbsDis')
    if os.path.exists(indir_proc) and os.path.exists(indir_level_acc) and os.path.exists(indir_level_dis):

        # Parse the model name
        match = re.match(r'([A-Z]*)_\d', modeldir)
        if match:
            modelname = match.group(1)
        else:
            modelname = modeldir

        # Read the azimuth of the longitudinal axis of the building (measured clockwise from North)
        azimuth_long_axis = np.genfromtxt(os.path.join(modelpath, modeldir, 'Info', 'azimuth_long_axis.txt'))

        # Load the blacklist if available
        blacklist_filename = os.path.join(recordpath, modeldir, 'blacklist.txt')
        if os.path.exists(blacklist_filename):
            blacklist = np.genfromtxt(blacklist_filename, dtype=str)
        else:
            blacklist = []

        # Obtain the list of sensor locations to be animated
        ani_locations = np.genfromtxt(os.path.join(modelpath, modeldir, 'Info', 'animation_sensors.txt'), \
                dtype=np.str)

        # Obtain the length of zero pad added as part of the record processing
        with open('process_records.py') as f:
            for line in f:
                match = re.match(r'filter_order = (.*)', line.strip())
                if match:
                    filter_order = int(match.group(1))

        with open('process_records.py') as f:
            for line in f:
                match = re.match(r'freq_corner_lower = (.*)', line.strip())
                if match:
                    freq_corner_lower = float(match.group(1))

        zero_pad_length = 1.5*filter_order/freq_corner_lower
        zero_pad_numpts = int(round(zero_pad_length/dt_int))

        # Define the output directory
        outdir = os.path.join(recordpath, modeldir, 'Animations')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Read the list of earthquakes in 'Raw' and the corresponding time steps
        # Use ravel() to account for the fact that the 'rec_info.txt' file could contain just one record, which
        # leads to a 0d NumPy array
        indir_raw = os.path.join(recordpath, modeldir, 'Raw')
        rec_data = np.genfromtxt(os.path.join(indir_raw, 'rec_info.txt'), dtype=None, usecols=(0, 1)).ravel()

        # Loop over all earthquakes
        for (earthquake, dt) in rec_data:
            print earthquake[:-4]

            # Import, interpolate, and rotate the horizontal base ground motions
            acc_base_int_trunc = {}
            dis_base_int_trunc = {}
            ymax = 0
            for direction in ('L', 'T', 'V'):
                if os.path.exists(os.path.join(indir_level_acc, '{}_{}_h0.th'.format(earthquake[:-4], \
                        direction))) and os.path.exists(os.path.join(indir_level_dis, \
                        '{}_{}_h0.th'.format(earthquake[:-4], direction))):

                    acc_base = np.genfromtxt(os.path.join(indir_level_acc, \
                            '{}_{}_h0.th'.format(earthquake[:-4], direction)))

                    if direction == 'L':
                        time = dt*np.arange(len(acc_base))
                        time_int = np.arange(dt_int, time[-1], dt_int)

                    acc_base_int = scipy.interpolate.interp1d(time, acc_base)(time_int)
                    acc_base_int_trunc[direction] = acc_base_int[zero_pad_numpts:-zero_pad_numpts]

                    if direction in ('L', 'T'):
                        ymax = max(ymax, max(abs(acc_base_int_trunc[direction])))

                    dis_base = np.genfromtxt(os.path.join(indir_level_dis, \
                            '{}_{}_h0.th'.format(earthquake[:-4], direction)))/100.0
                    dis_base_int = scipy.interpolate.interp1d(time, dis_base)(time_int)
                    dis_base_int_trunc[direction] = dis_base_int[zero_pad_numpts:-zero_pad_numpts]

            if 'V' not in dis_base_int_trunc and 'L' in dis_base_int_trunc and 'T' in dis_base_int_trunc:
                dis_base_int_trunc['V'] = np.zeros(len(dis_base_int_trunc['L']))
            elif 'L' not in dis_base_int_trunc or 'T' not in dis_base_int_trunc:
                print 'Base motions not available for {}'.format(earthquake[:-4])
                break

            azi_cos = np.cos(np.deg2rad(azimuth_long_axis) - np.pi/2.0)
            azi_sin = np.sin(np.deg2rad(azimuth_long_axis) - np.pi/2.0)
            rot_mat = np.array([[azi_cos, azi_sin], [-azi_sin, azi_cos]])
            dis_base_int_trunc_rot_lt = rot_mat.dot(np.array([dis_base_int_trunc['L'], dis_base_int_trunc['T']]))

            dis_base_int_trunc_rot = {'x':dis_base_int_trunc_rot_lt[0], 'y':dis_base_int_trunc_rot_lt[1], \
                    'z':dis_base_int_trunc['V']}

            # Parse the earthquake date
            eq_time = np.datetime64(datetime.datetime.strptime(re.match(r'(\d*\.\d*\.\d*).*', \
                    earthquake).group(1), '%Y.%j.%H%M'))

            # Read the metadata of all sensors in the building from the GeoNet DELTA Github repository
            # The sequence in which the sensors are listed in 'sites.csv' is the sequence in which their
            # recordings are listed in the *.cmf files
            sites = np.genfromtxt(os.path.join(inpath_delta, 'network', 'sites.csv'), dtype=None, names=True, \
                    delimiter=',', case_sensitive='lower', usecols=('station', 'location', 'latitude', \
                    'longitude', 'start_date', 'end_date'))

            idx = (sites['station'] == modelname)
            locations = sites['location'][idx]
            longitudes = sites['longitude'][idx]
            latitudes = sites['latitude'][idx]

            recorders = np.genfromtxt(os.path.join(inpath_delta, 'install', 'recorders.csv'), dtype=None, \
                    names=True, delimiter=',', case_sensitive='lower', usecols=('station', 'location', \
                    'azimuth', 'dip', 'depth', 'start_date', 'end_date'), \
                    converters={'start_date':date_time_conv, 'end_date':date_time_conv})
            sensors = np.genfromtxt(os.path.join(inpath_delta, 'install', 'sensors.csv'), dtype=None, \
                    names=True, delimiter=',', case_sensitive='lower', usecols=('station', 'location', \
                    'azimuth', 'dip', 'depth', 'start_date', 'end_date'), \
                    converters={'start_date':date_time_conv, 'end_date':date_time_conv})

            recorders_sensors = {}
            for key in recorders.dtype.fields:
                recorders_sensors[key] = np.hstack((recorders[key], sensors[key]))

            idx = ((recorders_sensors['station'] == modelname) & (eq_time >= recorders_sensors['start_date']) \
                    & (eq_time <= recorders_sensors['end_date']))
            azimuths = recorders_sensors['azimuth'][idx]
            dips = recorders_sensors['dip'][idx]
            depths = recorders_sensors['depth'][idx]
            start_dates = recorders_sensors['start_date'][idx]
            end_dates = recorders_sensors['end_date'][idx]

            sort_idx = []
            for location in locations:
                sort_idx += [np.flatnonzero(recorders_sensors['location'][idx] == location)[0]]
            sort_idx = np.array(sort_idx)
            azimuths = azimuths[sort_idx]
            dips = dips[sort_idx]
            depths = depths[sort_idx]
            start_dates = start_dates[sort_idx]
            end_dates = end_dates[sort_idx]

            # Import and rotate the displacement time histories recorded from sensors for which x and y
            # channels have been successfully processed. If z channel data is available, use it, else, treat
            # vertical movement to be zero.
            proc_rec_data = np.genfromtxt(os.path.join(indir_proc, 'rec_info.txt'), dtype=None, usecols=(0,))
            ani_sensors = []
            dis_rel_int_trunc_rot = []
            for i in range(len(locations)):
                if '{}_{:02d}x.th'.format(earthquake[:-4], i + 1) in proc_rec_data and \
                        '{}_{:02d}y.th'.format(earthquake[:-4], i + 1) in proc_rec_data and \
                        '{}_{:02d}x.th'.format(earthquake[:-4], i + 1) not in blacklist and \
                        '{}_{:02d}y.th'.format(earthquake[:-4], i + 1) not in blacklist and \
                        locations[i] in ani_locations:
                    ani_sensors += [i + 1]
                    dis_int_trunc = {}
                    for direction in ('x', 'y'):
                        dis = np.genfromtxt(os.path.join(indir_proc, '{}_{:02d}{}.th'.format(earthquake[:-4], \
                                i + 1, direction)))/100.0
                        dis_int = scipy.interpolate.interp1d(time, dis)(time_int)
                        dis_int_trunc[direction] = dis_int[zero_pad_numpts:-zero_pad_numpts]
                    if '{}_{:02d}z.th'.format(earthquake[:-4], i + 1) in proc_rec_data and \
                            '{}_{:02d}z.th'.format(earthquake[:-4], i + 1) not in blacklist:
                        dis = np.genfromtxt(os.path.join(indir_proc, '{}_{:02d}z.th'.format(earthquake[:-4], \
                                i + 1)))/100.0
                        dis_int = scipy.interpolate.interp1d(time, dis)(time_int)
                        dis_int_trunc['z'] = dis_int[zero_pad_numpts:-zero_pad_numpts]
                    else:
                        dis_int_trunc['z'] = dis_base_int_trunc_rot['z']

                    azi_cos = np.cos(np.deg2rad(azimuths[i]))
                    azi_sin = np.sin(np.deg2rad(azimuths[i]))
                    rot_mat = np.array([[azi_cos, azi_sin], [-azi_sin, azi_cos]])
                    dis_int_trunc_rot_lt = rot_mat.dot(np.array([dis_int_trunc['x'], dis_int_trunc['y']]))

                    dis_rel_int_trunc_rot += [{'x':dis_int_trunc_rot_lt[0] - dis_base_int_trunc_rot['x'], \
                            'y':dis_int_trunc_rot_lt[1] - dis_base_int_trunc_rot['y'], \
                            'z':dis_int_trunc['z'] - dis_base_int_trunc_rot['z']}]

            # Compute the coordinates of all sensors
            # Use the coordinates of the first sensor as the origin
            ani_lon = []
            ani_lat = []
            ani_z = []
            for ani_sensor in ani_sensors:
                ani_lon += [longitudes[ani_sensor - 1]]
                ani_lat += [latitudes[ani_sensor - 1]]
                ani_z += [-depths[ani_sensor - 1]]

            lat_orig = np.deg2rad(ani_lat[0])
            delta_x = 111412.84*np.cos(lat_orig) - 93.5*np.cos(3.0*lat_orig) + 0.118*np.cos(5.0*lat_orig)
            delta_y = 111132.92 - 559.82*np.cos(2.0*lat_orig) + 1.175*np.cos(4.0*lat_orig) - \
                    0.0023*np.cos(6.0*lat_orig)

            ani_x = []
            ani_y = []
            for i in range(len(ani_lat)):
                ani_x += [delta_x*(ani_lon[i] - ani_lon[0])]
                ani_y += [delta_y*(ani_lat[i] - ani_lat[0])]

            # Initialize the plot
            fig = plt.figure(figsize=(8, 8))
            ax1 = plt.axes((0.05, 0.32, 0.90, 0.66), projection='3d')
            ax2 = plt.axes((0.11, 0.20, 0.84, 0.10))
            ax3 = plt.axes((0.11, 0.08, 0.84, 0.10))

            plt.rcParams['font.size'] = 12.0

            #ax1.axis('off')
            ax1.set_xticks(())
            ax1.set_yticks(())
            ax1.set_zticks(())
            ax1.set_title(r'Deformations scaled ${:.0f}x$'.format(scale[modeldir]))

            max_range = max([max(ani_x) - min(ani_x), max(ani_y) - min(ani_y), max(ani_z) - min(ani_z)])/2.0
            mid_x = (min(ani_x) + max(ani_x))/2.0
            mid_y = (min(ani_y) + max(ani_y))/2.0
            ax1.set_xlim(mid_x - max_range - hpad, mid_x + max_range + hpad)
            ax1.set_ylim(mid_y - max_range - hpad, mid_y + max_range + hpad)
            ax1.set_zlim(min(ani_z), max(ani_z) + vpad)

            time_int_plot = time_int[zero_pad_numpts:-zero_pad_numpts] - time_int[zero_pad_numpts]

            ax2.plot(time_int_plot, acc_base_int_trunc['L'], 'k', lw=0.4, zorder=0)
            ax2.set_xticks(())
            ax2.set_ylabel('$a_L$ ($g$)')
            ax2.set_xlim(0, time_int_plot[-1])
            ax2.set_ylim(-1.1*ymax, 1.1*ymax)
            ax2.grid(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.yaxis.set_ticks_position('left')

            ax3.plot(time_int_plot, acc_base_int_trunc['T'], 'k', lw=0.4, zorder=0)
            ax3.set_xlabel('$t$ ($s$)')
            ax3.set_ylabel('$a_T$ ($g$)')
            ax3.set_xlim(0, time_int_plot[-1])
            ax3.set_ylim(-1.1*ymax, 1.1*ymax)
            ax3.grid(False)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.xaxis.set_ticks_position('bottom')
            ax3.yaxis.set_ticks_position('left')

            lines = []

            # Initialize the points moving over the accelerogram
            lines += ax2.plot([], [], color='r', marker='o', markeredgecolor='r', markersize=3, lw=0.4, \
                    zorder=1)
            lines += ax3.plot([], [], color='r', marker='o', markeredgecolor='r', markersize=3, lw=0.4, \
                    zorder=1)

            # Initialize the sensors
            for ani_sensor in ani_sensors:
                lines += ax1.plot([], [], [], color='k', marker='o', markersize=6, zorder=1)
                
            # Function to plot one frame of the animation
            def plot_frame(frame_num):
                i = 0

                # Plot the points moving over the accelerograms
                lines[i].set_data(time_int_plot[frame_num], acc_base_int_trunc['L'][frame_num])
                i += 1
                lines[i].set_data(time_int_plot[frame_num], acc_base_int_trunc['T'][frame_num])
                i += 1

                for j in range(len(ani_sensors)):
                    lines[i].set_data(ani_x[j] + scale[modeldir]*dis_rel_int_trunc_rot[j]['x'][frame_num], \
                            ani_y[j] + scale[modeldir]*dis_rel_int_trunc_rot[j]['y'][frame_num])
                    lines[i].set_3d_properties(ani_z[j] + \
                            scale[modeldir]*dis_rel_int_trunc_rot[j]['z'][frame_num])
                    i += 1

                return lines

            # Animate the analysis results
            num_frames = len(time_int_plot)
            animation = ani.FuncAnimation(fig, plot_frame, frames=num_frames, blit=True, repeat=False)
            writer = ani.AVConvWriter(fps=fps, bitrate=bitrate)
            animation.save(os.path.join(outdir, '{}.mp4'.format(earthquake[:-4])), writer=writer)

#            num_frames = 2000
#            plot_frame(num_frames - 1)
#            plt.show()

    print
