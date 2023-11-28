import laspy
import io

def compress_lidar_frame(lidars_i):
    # LiDAR is hard to compress so we use a special purpose format.
    header = laspy.LasHeader(point_format=self.config.point_format)
    header.offsets = np.min(lidars_i, axis=0)
    header.scales = np.array(
        [self.config.point_precision, self.config.point_precision, self.config.point_precision])
    compressed_lidar_i = io.BytesIO()
    with laspy.open(compressed_lidar_i, mode='w', header=header, do_compress=True, closefd=False) as writer:
        point_record = laspy.ScaleAwarePointRecord.zeros(lidars_i.shape[0], header=header)
        point_record.x = lidars_i[:, 0]
        point_record.y = lidars_i[:, 1]
        point_record.z = lidars_i[:, 2]
        writer.write_points(point_record)

    compressed_lidar_i.seek(0)  # Resets file handle to the start

    return compressed_lidar_i