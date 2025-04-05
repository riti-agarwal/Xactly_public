import React, { useState, useEffect } from 'react';
import { Box, styled, Typography, Tooltip } from '@mui/material';
import Masonry from 'react-masonry-css';

const MasonryContainer = styled(Box)({
  width: '100%',
  '.my-masonry-grid': {
    display: 'flex',
    marginLeft: '-16px', /* gutter size offset */
    width: 'auto',
  },
  '.my-masonry-grid_column': {
    paddingLeft: '16px', /* gutter size */
    backgroundClip: 'padding-box',
  },
});

const ProductCard = styled(Box)({
  backgroundColor: '#2d2d2d',
  borderRadius: '8px',
  overflow: 'hidden',
  marginBottom: '16px',
  transition: 'transform 0.2s, box-shadow 0.2s',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: '0 10px 20px rgba(0,0,0,0.2)',
  },
});

const ProductImage = styled('img')({
  width: '100%',
  height: 'auto',
  display: 'block',
  cursor: 'grab',
  borderRadius: '8px 8px 0 0',
  '&:active': {
    cursor: 'grabbing',
  }
});

const MetricsContainer = styled(Box)({
  padding: '8px',
  backgroundColor: 'rgba(0, 0, 0, 0.7)',
  borderRadius: '0 0 8px 8px',
});

const MetricRow = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  marginBottom: '4px',
  '&:last-child': {
    marginBottom: 0,
  }
});

const MetricLabel = styled(Typography)({
  color: 'white',
  fontSize: '10px',
  width: '90px',
  flexShrink: 0,
});

const MetricBar = styled(Box)({
  flex: 1,
  height: '4px',
  backgroundColor: '#333',
  borderRadius: '2px',
  overflow: 'hidden',
});

const MetricFill = styled(Box)({
  height: '100%',
  borderRadius: '2px',
});

const ProductGrid = () => {
  const [products, setProducts] = useState([]);
  const [loaded, setLoaded] = useState(false);
  
  // Responsive breakpoints for the masonry layout
  const breakpointColumnsObj = {
    default: 4,
    1100: 3,
    700: 2,
    500: 1
  };

  useEffect(() => {
    // Generate random metrics for each product
    const generateRandomMetrics = () => {
      return {
        quality: Math.floor(Math.random() * 51) + 50, // Random value between 50-100
        history_quality: Math.floor(Math.random() * 51) + 50,
        trend_quality: Math.floor(Math.random() * 51) + 50
      };
    };

    // Using a mix of portrait and landscape images with randomized quality metrics
    const shoeImages = [
      { 
        id: 1, 
        imageUrl: 'https://images.unsplash.com/photo-1542291026-7eec264c27ff?q=80&w=600&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 2, 
        imageUrl: 'https://images.unsplash.com/photo-1608231387042-66d1773070a5?q=80&w=800&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 3, 
        imageUrl: 'https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?q=80&w=700&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 4, 
        imageUrl: 'https://images.unsplash.com/photo-1600185365926-3a2ce3cdb9eb?q=80&w=500&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 5, 
        imageUrl: 'https://images.unsplash.com/photo-1551107696-a4b0c5a0d9a2?q=80&w=600&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 6, 
        imageUrl: 'https://images.unsplash.com/photo-1460353581641-37baddab0fa2?q=80&w=900&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 7, 
        imageUrl: 'https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?q=80&w=400&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 8, 
        imageUrl: 'https://images.unsplash.com/photo-1605348532760-6753d2c43329?q=80&w=700&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 9, 
        imageUrl: 'https://images.unsplash.com/photo-1491553895911-0055eca6402d?q=80&w=500&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 10, 
        imageUrl: 'https://images.unsplash.com/photo-1560769629-975ec94e6a86?q=80&w=600&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 11, 
        imageUrl: 'https://images.unsplash.com/photo-1543508282-6319a3e2621f?q=80&w=800&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 12, 
        imageUrl: 'https://images.unsplash.com/photo-1600269452121-4f2416e55c28?q=80&w=550&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 13, 
        imageUrl: 'https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?q=80&w=600&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 14, 
        imageUrl: 'https://images.unsplash.com/photo-1512374382149-233c42b6a83b?q=80&w=750&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
      { 
        id: 15, 
        imageUrl: 'https://images.unsplash.com/photo-1518002171953-a080ee817e1f?q=80&w=450&auto=format&fit=crop',
        ...generateRandomMetrics()
      },
    ];
    setProducts(shoeImages);
    setLoaded(true);
  }, []);

  const handleImageLoad = (e) => {
    e.target.style.opacity = 1;
  };

  return (
    <MasonryContainer>
      <Masonry
        breakpointCols={breakpointColumnsObj}
        className="my-masonry-grid"
        columnClassName="my-masonry-grid_column"
      >
        {loaded && products.map((product) => (
          <ProductCard key={product.id}>
            <Box>
              <ProductImage 
                src={product.imageUrl}
                alt={`Product ${product.id}`}
                onLoad={handleImageLoad}
                style={{ opacity: 0, transition: 'opacity 0.3s' }}
                draggable="true"
                onDragStart={(e) => {
                  e.dataTransfer.setData('text/plain', product.imageUrl);
                  e.dataTransfer.effectAllowed = 'copy';
                }}
              />
              <MetricsContainer>
                <Tooltip title={`Quality: ${product.quality}%`} placement="right">
                  <MetricRow>
                    <MetricLabel>Quality</MetricLabel>
                    <MetricBar>
                      <MetricFill 
                        sx={{ 
                          width: `${product.quality}%`,
                          backgroundColor: '#3f51b5' // Indigo
                        }} 
                      />
                    </MetricBar>
                  </MetricRow>
                </Tooltip>
                
                <Tooltip title={`Historical Quality: ${product.history_quality}%`} placement="right">
                  <MetricRow>
                    <MetricLabel>Historical</MetricLabel>
                    <MetricBar>
                      <MetricFill 
                        sx={{ 
                          width: `${product.history_quality}%`,
                          backgroundColor: '#009688' // Teal
                        }} 
                      />
                    </MetricBar>
                  </MetricRow>
                </Tooltip>
                
                <Tooltip title={`Trend Quality: ${product.trend_quality}%`} placement="right">
                  <MetricRow>
                    <MetricLabel>Trend</MetricLabel>
                    <MetricBar>
                      <MetricFill 
                        sx={{ 
                          width: `${product.trend_quality}%`,
                          backgroundColor: '#e91e63' // Pink
                        }} 
                      />
                    </MetricBar>
                  </MetricRow>
                </Tooltip>
              </MetricsContainer>
            </Box>
          </ProductCard>
        ))}
      </Masonry>
    </MasonryContainer>
  );
};

export default ProductGrid;
