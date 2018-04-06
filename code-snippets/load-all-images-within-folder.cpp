namespace aia
{
	// retrieves and loads all images within the given folder and having the given file extension
	std::vector < cv::Mat > getImagesInFolder(std::string folder, std::string ext = ".tif", bool force_gray = false) throw (aia::error)
	{
		// check folders exist
		if(!ucas::isDirectory(folder))
			throw aia::error(aia::strprintf("in getImagesInFolder(): cannot open folder at \"%s\"", folder.c_str()));

		// get all files within folder
		std::vector < std::string > files;
		cv::glob(folder, files);

		// open files that contains 'ext'
		std::vector < cv::Mat > images;
		for(auto & f : files)
		{
			if(f.find(ext) == std::string::npos)
				continue;

			images.push_back(cv::imread(f, force_gray ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_UNCHANGED));
		}

		return images;
	}
}