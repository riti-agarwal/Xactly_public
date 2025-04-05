import React, { useState, useEffect } from 'react';
import { Box, styled, Typography, Tooltip, CircularProgress } from '@mui/material';
import Masonry from 'react-masonry-css';
import { fetchRandomImages } from '../services/apis';
import { useSidebar } from '../context/SidebarContext';

const MasonryContainer = styled(Box)(props => ({
  width: '100%',
  transition: 'padding-left 0.3s ease-in-out',
  paddingLeft: props.sidebarVisible ? '300px' : '0',
  '.my-masonry-grid': {
    display: 'flex',
    marginLeft: '-16px', /* gutter size offset */
    width: 'auto',
  },
  '.my-masonry-grid_column': {
    paddingLeft: '16px', /* gutter size */
    backgroundClip: 'padding-box',
  },
}));

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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { isSidebarVisible } = useSidebar();
  
  // Responsive breakpoints for the masonry layout
  const breakpointColumnsObj = {
    default: 4,
    1100: 3,
    700: 2,
    500: 1
  };

  useEffect(() => {
    const loadProducts = async () => {
      try {
        setLoading(true);
        const data = await fetchRandomImages();
        
        // Map the API response to our product format
        const formattedProducts = data.images.map(item => ({
          imageUrl: item.url,
          quality: item.quality,
          history_quality: item.history_quality,
          trend_quality: item.trend_quality
        }));
        
        setProducts(formattedProducts);
        setError(null);
      } catch (err) {
        console.error('Failed to load products:', err);
        setError('Failed to load products. Please try again later.');
      } finally {
        setLoading(false);
        setLoaded(true);
      }
    };

    loadProducts();
  }, []);

  const handleImageLoad = (e) => {
    e.target.style.opacity = 1;
  };

  return (
    <MasonryContainer sidebarVisible={isSidebarVisible}>
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', width: '100%', padding: '40px' }}>
          <CircularProgress sx={{ color: '#fff' }} />
        </Box>
      )}
      
      {error && (
        <Box sx={{ color: '#ff5252', textAlign: 'center', width: '100%', padding: '20px' }}>
          <Typography variant="body1">{error}</Typography>
        </Box>
      )}
      
      <Masonry
        breakpointCols={breakpointColumnsObj}
        className="my-masonry-grid"
        columnClassName="my-masonry-grid_column"
      >
        {loaded && products.map((product) => (
          <ProductCard key={product.id}>
            <Box>
              <ProductImage 
                src={`data:image/jpeg;base64,${product.imageUrl}`}
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
