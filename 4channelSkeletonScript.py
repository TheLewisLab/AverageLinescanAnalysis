#@ File (label="Select an image directory", style="directory") imagedir
#@ File (label="Select a save directory", style="directory") savedir
#@ File (label="Select reference image for pixel data", style="file") ref_file
from ij.gui import Roi, PolygonRoi
from ij import IJ, WindowManager, ImagePlus
from ij.gui import ProfilePlot
from ij.measure import ResultsTable
from ij.plugin.frame import RoiManager
from sc.fiji.analyzeSkeleton import AnalyzeSkeleton_
from java.awt import Polygon
import os, glob, sys

branch_threshold = 8
iterations = 5
linescan_threshold = 30
stroke_width = 5
pixel_data = IJ.openImage(ref_file.getPath()).getCalibration()

def threshold_image(image):
	thresh_image = image.duplicate()
	IJ.setThreshold(thresh_image, 1, 65536)
	
	return thresh_image
		
def filter_length(input_img, thr, iters):
	og_img = input_img
	img = input_img.duplicate()
	thresh = thr
	for its in range(0, iters):
		skel = AnalyzeSkeleton_()
		skel.setup("", img)
		results = skel.run(0, False, False, None, True, True)
		outStack = img.getStack()
		graph = results.getGraph()
		end_points = results.getListOfEndPoints()
		
		print(len(end_points))
		
		for i in range(0, len(graph)):
			list_edges = graph[i].getEdges()
			for e in list_edges:
				p = e.getV1().getPoints()
				v1_end = end_points.contains(p.get(0))
				p2 = e.getV2().getPoints()
				v2_end = end_points.contains(p2.get(0))
				if (v1_end or v2_end):
					if e.getLength() < (thresh*img.getCalibration().pixelWidth):
						if v1_end:
							outStack.setVoxel(p.get(0).x, p.get(0).y, p.get(0).z, 0)
						if v2_end:
							outStack.setVoxel(p2.get(0).x, p2.get(0).y, p2.get(0).z, 0)
						for p in e.getSlabs():
							outStack.setVoxel(p.x, p.y, p.z, 0)
				if (e.getLength() < (2*img.getCalibration().pixelWidth)):
					for p in e.getSlabs():
						outStack.setVoxel(p.x, p.y, p.z, 0)
		new_image = img.duplicate()
		img.close()
		img = new_image
	junctions = results.getListOfJunctionVoxels()
	temp_stack = img.getImageStack()
	for j in junctions:
		temp_stack.setVoxel(j.x, j.y, j.z, 0)
	return img

def add_lines_to_roi(final_image):
	skel = AnalyzeSkeleton_()
	skel.setup("", final_image)
	results = skel.run(0, False, True, None, True, False)
	lines = skel.getShortestPathPoints()
	RM = RoiManager()
	roi_manager = RM.getRoiManager()
	for l in lines:
		if len(l) > linescan_threshold:
			polygon = Polygon()
			for cord in l:
				polygon.addPoint(cord.x, cord.y)
			poly_line = PolygonRoi(polygon, Roi.POLYLINE)
			poly_line.setStrokeWidth(stroke_width)
			roi_manager.addRoi(poly_line)

def save_roi_line_scans(image):
	RM = RoiManager()
	roi_manager = RM.getRoiManager()
	image.show()
	table = ResultsTable()
	for i in range(0, roi_manager.getCount()):
		roi_manager.select(i)
		# Set distance column once per ROI
		image.setSlice(1)  # Use first slice to get profile length
		temp_profile = ProfilePlot(image)
		profile_length = len(temp_profile.getProfile())
		table.setValues('ROI_'+ str(i) +' distance ('+str(pixel_data.getUnit())+')', [k*pixel_data.pixelWidth for k in range(profile_length)])
		
		# Now get intensity profiles for each channel
		for j in range(0, image.getStack().getSize()):
			image.setSlice(j + 1)
			profile = ProfilePlot(image)
			profile = profile.getProfile()
			table.setValues('ROI_'+ str(i) + ' Channel_'+str(j+1), profile)
			#table.updateResults()
	table.save(savedir.getPath() + '/' + image.getShortTitle() + '.csv')
	roi_manager.close()

image_root = imagedir.getPath()
for files in os.listdir(image_root):
	if files.endswith('.tiff'):
		image = IJ.openImage(image_root + '/' + files)
		c1 = IJ.openImage(image_root + '/' + files, 1)
		c2 = IJ.openImage(image_root + '/' + files, 2)
		c3 = IJ.openImage(image_root + '/' + files, 3)
		c4 = IJ.openImage(image_root + '/' + files, 4)  # Added 4th channel
		print(image.getTitle())
		thresh_image = threshold_image(c1)
		IJ.run(thresh_image, 'Convert to Mask', '')
		IJ.run(thresh_image, 'Skeletonize (2D/3D)', '')
		# need to pass thresholded image here
		skel_image = filter_length(thresh_image, branch_threshold, iterations)
		
		add_lines_to_roi(skel_image)
		save_roi_line_scans(image)
		image.close()
		thresh_image.close()
		skel_image.close()
		c1.close()
		c2.close()
		c3.close()
		c4.close()  # Added 4th channel close