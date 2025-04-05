import React, { useState, useRef } from 'react';
import { Box, Typography, styled, IconButton } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import CloseIcon from '@mui/icons-material/Close';

const CollectionsContainer = styled(Box)({
  backgroundColor: '#2d2d2d',
  borderRight: '1px solid #333',
  position: 'absolute',
  left: '-300px',
  top: 0,
  bottom: 0,
  width: '300px',
  zIndex: 5,
  display: 'flex',
  flexDirection: 'column',
  transition: 'left 0.3s ease-in-out',
  '&.visible': {
    left: '0',
  }
});

const SidebarHeader = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '16px',
  borderBottom: '1px solid #333',
});

const SectionTitle = styled(Typography)({
  color: 'white',
  fontWeight: 500,
  marginBottom: '12px',
});

const SectionContainer = styled(Box)({
  padding: '16px',
  borderBottom: '1px solid #333',
  flex: 1,
  overflowY: 'auto',
});

const ImagesContainer = styled(Box)({
  display: 'flex',
  flexWrap: 'wrap',
  gap: '12px',
  marginTop: '12px',
});

const ImagePlaceholder = styled(Box)({
  width: '100px',
  height: '100px',
  backgroundColor: '#1a1a1a',
  borderRadius: '4px',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  border: '2px dashed #555',
  flexShrink: 0,
});

const ImageContainer = styled(Box)({
  position: 'relative',
  width: '100px',
  height: '100px',
  borderRadius: '4px',
  overflow: 'hidden',
  '&:hover .remove-button': {
    opacity: 1,
  }
});

const CollectionImage = styled('img')({
  width: '100%',
  height: '100%',
  objectFit: 'cover',
  borderRadius: '4px',
  flexShrink: 0,
});

const RemoveButton = styled(IconButton)({
  position: 'absolute',
  top: 0,
  right: 0,
  backgroundColor: 'rgba(0, 0, 0, 0.5)',
  color: 'white',
  padding: '2px',
  opacity: 0,
  transition: 'opacity 0.2s',
  '&:hover': {
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
  },
  '& .MuiSvgIcon-root': {
    fontSize: '16px',
  }
});

const DropZone = styled(Box)({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundColor: 'rgba(0, 120, 212, 0.2)',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  zIndex: 10,
  pointerEvents: 'none',
  opacity: 0,
  transition: 'opacity 0.3s',
  '&.active': {
    opacity: 1,
  }
});

const ProductCollections = () => {
  // Reference to the container for click outside detection
  const containerRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);
  const [historicImages, setHistoricImages] = useState([]);
  const [trendImages, setTrendImages] = useState([]);
  const [isDraggingOverHistoric, setIsDraggingOverHistoric] = useState(false);
  const [isDraggingOverTrends, setIsDraggingOverTrends] = useState(false);

  const handleHistoricDragOver = (e) => {
    e.preventDefault();
    setIsDraggingOverHistoric(true);
  };

  const handleTrendsDragOver = (e) => {
    e.preventDefault();
    setIsDraggingOverTrends(true);
  };

  const handleDragLeave = (section) => {
    if (section === 'historic') {
      setIsDraggingOverHistoric(false);
    } else {
      setIsDraggingOverTrends(false);
    }
  };

  const handleDrop = (e, section) => {
    e.preventDefault();
    const imageUrl = e.dataTransfer.getData('text/plain');
    
    if (imageUrl) {
      if (section === 'historic') {
        setHistoricImages([...historicImages, imageUrl]);
        setIsDraggingOverHistoric(false);
      } else {
        setTrendImages([...trendImages, imageUrl]);
        setIsDraggingOverTrends(false);
      }
    }
  };

  const removeImage = (section, index) => {
    if (section === 'historic') {
      const newImages = [...historicImages];
      newImages.splice(index, 1);
      setHistoricImages(newImages);
    } else {
      const newImages = [...trendImages];
      newImages.splice(index, 1);
      setTrendImages(newImages);
    }
  };

  // Toggle visibility of the entire collections container
  const toggleVisibility = () => {
    setIsVisible(!isVisible);
  };

  // We've removed the click-outside behavior to allow for easier drag and drop operations

  return (
    <>
      <Box 
        data-collections-toggle="true"
        sx={{
          position: 'absolute',
          left: isVisible ? '300px' : '0',
          top: '50%',
          transform: 'translateY(-50%)',
          backgroundColor: '#1a1a1a',
          borderRadius: '0 4px 4px 0',
          cursor: 'pointer',
          zIndex: 6,
          transition: 'left 0.3s ease-in-out',
          boxShadow: '2px 0 5px rgba(0,0,0,0.2)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '32px',
          height: '64px',
        }} 
        onClick={toggleVisibility}
      >
        <IconButton size="small" sx={{ color: 'white', padding: 0 }}>
          {isVisible ? <ChevronLeftIcon /> : <ExpandMoreIcon sx={{ transform: 'rotate(-90deg)' }} />}
        </IconButton>
      </Box>
      <CollectionsContainer ref={containerRef} className={isVisible ? 'visible' : ''}>        
        <SectionContainer
          sx={{ position: 'relative' }}
          onDragOver={handleHistoricDragOver}
          onDragLeave={() => handleDragLeave('historic')}
          onDrop={(e) => handleDrop(e, 'historic')}
        >
          <SectionTitle variant="h6">
            Historic Purchase
          </SectionTitle>
          
          {isDraggingOverHistoric && (
            <DropZone className="active">
              <Typography variant="body1" sx={{ color: 'white', backgroundColor: 'rgba(0,0,0,0.7)', padding: '8px 16px', borderRadius: '4px' }}>
                Drop to add to Historic Purchase
              </Typography>
            </DropZone>
          )}
          
          <ImagesContainer>
            {historicImages.length > 0 ? (
              historicImages.map((img, index) => (
                <ImageContainer key={index}>
                  <CollectionImage src={img} alt={`Historic item ${index}`} />
                  <RemoveButton 
                    className="remove-button"
                    size="small" 
                    onClick={() => removeImage('historic', index)}
                  >
                    <CloseIcon />
                  </RemoveButton>
                </ImageContainer>
              ))
            ) : (
              <>
                <ImagePlaceholder>
                  <Typography variant="body2" sx={{ color: '#888', textAlign: 'center' }}>
                    Drag items here
                  </Typography>
                </ImagePlaceholder>
                <ImagePlaceholder>
                  <Typography variant="body2" sx={{ color: '#888', textAlign: 'center' }}>
                    Drag items here
                  </Typography>
                </ImagePlaceholder>
              </>
            )}
          </ImagesContainer>
        </SectionContainer>

        <SectionContainer
          sx={{ position: 'relative' }}
          onDragOver={handleTrendsDragOver}
          onDragLeave={() => handleDragLeave('trends')}
          onDrop={(e) => handleDrop(e, 'trends')}
        >
          <SectionTitle variant="h6">
            Trends
          </SectionTitle>
          
          {isDraggingOverTrends && (
            <DropZone className="active">
              <Typography variant="body1" sx={{ color: 'white', backgroundColor: 'rgba(0,0,0,0.7)', padding: '8px 16px', borderRadius: '4px' }}>
                Drop to add to Trends
              </Typography>
            </DropZone>
          )}
          
          <ImagesContainer>
            {trendImages.length > 0 ? (
              trendImages.map((img, index) => (
                <ImageContainer key={index}>
                  <CollectionImage src={img} alt={`Trend item ${index}`} />
                  <RemoveButton 
                    className="remove-button"
                    size="small" 
                    onClick={() => removeImage('trends', index)}
                  >
                    <CloseIcon />
                  </RemoveButton>
                </ImageContainer>
              ))
            ) : (
              <>
                <ImagePlaceholder>
                  <Typography variant="body2" sx={{ color: '#888', textAlign: 'center' }}>
                    Drag items here
                  </Typography>
                </ImagePlaceholder>
                <ImagePlaceholder>
                  <Typography variant="body2" sx={{ color: '#888', textAlign: 'center' }}>
                    Drag items here
                  </Typography>
                </ImagePlaceholder>
              </>
            )}
          </ImagesContainer>
        </SectionContainer>
    </CollectionsContainer>
    </>
  );
};

export default ProductCollections;
