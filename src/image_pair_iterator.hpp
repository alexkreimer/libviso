class image_iterator
    : public boost::iterator_facade<image_iterator,
				    cv::Mat,
				    boost::forward_traversal_tag>
{
public:
    image_iterator() : m_index(-1) {}
    explicit image_iterator(const std::string& mask,
			    int index_begin=0,
			    int index_end=-1) 
	: m_mask(mask), m_index(index_begin-1), m_index_end(index_end),
	  m_image(0) 
	{
	    this->increment();
	    std::cout << "image_iterator created" << std::endl;
	}
private:
    friend class boost::iterator_core_access;
    void increment() { 
	this->m_index++;
	std::cout << "image_iterator increment" << std::endl;
	std::string file_name = str(boost::format(m_mask) % m_index);
	if (!boost::filesystem::exists(file_name)) {
	    delete m_image;
	    m_image = 0;
	    m_index = -1;
	} else {
	    delete m_image;
	    m_image = new cv::Mat(imread(file_name, IMREAD_GRAYSCALE));
	}
    }

    bool equal(image_iterator const &other) const {
	return this->m_index == other.m_index && this->m_mask == other.m_mask;
    }
    
    cv::Mat& dereference() const {
	assert(m_index >= 0);
	return *m_image ;
    }

    int m_index;
    int m_index_end;
    std::string m_mask;
    cv::Mat *m_image;
};
