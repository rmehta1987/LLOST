function [newcellTumor] = convert_to_cloud(thetumor) 
%This script creates isosurfaces for Lung Adenocarcinoma Lesions in the workspace data
%theoriginalTumors_LUAD.mat .  This mat file loads a struct of size 41 lesions which
%also includes %contains the original data (var name: orgdata), the delianted data(var name: data), 
%slices where lesions exist in the orginal dicom (var name: slice) 
%slice thickness (var name: thickness), patient-barcode (var name: patientname),
%number of slices (var name: num_slices), and dicom-info of each slice (var name: allSliceDcminfo)

%We follow this paradigm:
%Interpolate slices
%Create a meshgrid, where we interpolate along the Z-axis
% [x,y,z] = size(TUMOR_DATA);
% [X,Y,Z]= meshgrid(1:x,1:y,1:z); % Original Coordinates -- Don't really need
% [X1,Y1,Z1]= meshgrid(1:x,1:y,1:.2:z); % Interpolation (X and Y coords
% remain the same), where .2 is 1/slicethickness
% 
% %Get coordinates of original slices where data was 1 (tumor exists)
% [X2,Y2,Z2] = ind2sub(size(lesion.data),find(lesion.data > 0));
% data = ones(size(X2,1),1); % make vector of ones, that correspond to where coordinates of lesion
% F = griddata(X2(:),Y2(:),Z2(:),data,X1,Y1,Z1); % interpolate
% [X3,Y3,Z3] = ind2sub(size(F),find(F > 0));
%  % Get the coordinates created from the meshgrid from the interpolated version
%  % Interpolate creates a new slice, but we need the coordinates to match so
%  % for example if 1:.2:z created 46 new slices, the interpolation will be
%  % where .2 is the 1/slice_thickness
%  % of dim 512x512x46, but we need the coordinates of the Z, and not just 46
%  % slices, so Z(:,:,46) should map to 10, and Z(:,:,45) map to 9.8
% 
% Returns an updated cell-array with var name: isosurface containing point 
% cloud coordinates along with a pointcloud for plotting purposes
% 


num_tumors = size(thetumor,1);
newcellTumor=thetumor;

for i=1:num_tumors
    % prev_data = thetumor{i}.data; % contour data
    %%%%% Transform data into real-world coordinates %%%%
    
    if length(thetumor{i}.slice) > 1
        transmatrix = convert_to_real_world(thetumor{i}); % transformation matrix
        newcellTumor{i}.transmatrix = transmatrix;

        %%%% Next Convert to Point cloud %%%%
        data = thetumor{i}.data;
        slice_thick = thetumor{i}.thickness; % slice thickness
        [x,y,z] = size(data);
        between_slice = 1/slice_thick;
        [X1,Y1,Z1]= meshgrid(1:x,1:y,1:between_slice:z); % Interpolation (X and Y coords remain the same)
        [X2,Y2,Z2] = ind2sub(size(data),find(data > 0)); %Get coordinates of original slices where data was 1 (tumor exists)
        data = ones(size(X2,1),1); % make vector of ones, that correspond to where coordinates of lesion
        F = griddata(X2(:),Y2(:),Z2(:),data,X1,Y1,Z1); % interpolate
        [X3,Y3,Z3] = ind2sub(size(F),find(F > 0));
        Z3_1(:) = Z1(1,1,Z3(:)); % because Z3 are indices of the matrix, we want the points on Z-axis 
        coords = [X3(:),Y3(:),Z3_1(:)]'; % dims 3 x num_coords
        newcellTumor{i}.imageIsoSurface = ones(size(coords,2),3);
        newcellTumor{i}.imageIsoSurface(:,1) = coords(1,:)';
        newcellTumor{i}.imageIsoSurface(:,2) = coords(2,:)';
        newcellTumor{i}.imageIsoSurface(:,3) = coords(3,:)';
        coords(4,1:length(Y3))=1; % add extra dimension of 1's for transformation 
        newCoord = transmatrix*coords;
        finalCoord = [newCoord(1,:); newCoord(2,:); newCoord(3,:)]'; % get rid of last dimension
        ptCloud = pointCloud(finalCoord); % convert to ptCloud
        cmatrix = ones(size(ptCloud.Location)).*[1 0 0];
        ptCloud = pointCloud(finalCoord,'Color',cmatrix); % for plotting purposes
        the_coords = reshape(ptCloud.Location, [], 3);

        newcellTumor{i}.Patientisosurface=ones(size(the_coords,1),3);
        newcellTumor{i}.Patientisosurface(:,1) = the_coords(:,1);
        newcellTumor{i}.Patientisosurface(:,2) = the_coords(:,2);
        newcellTumor{i}.Patientisosurface(:,3) = the_coords(:,3);
        clear Z3_1;
    else
        data = thetumor{i}.data;
        [X2,Y2,Z2] = ind2sub(size(data),find(data > 0));
        transmatrix = convert_to_real_world(thetumor{i}); % transformation matrix
        newcellTumor{i}.transmatrix = transmatrix;
        coords = [X2(:),Y2(:),Z2(:)]';
        newcellTumor{i}.imageIsoSurface = ones(size(coords,1),3);
        newcellTumor{i}.imageIsoSurface(:,1) = coords(:,1);
        newcellTumor{i}.imageIsoSurface(:,2) = coords(:,2);
        newcellTumor{i}.isosurface(:,3) = coords(:,3);
        coords(4,1:length(Y2))=1;
        newCoord = transmatrix*coords;
        finalCoord = [newCoord(1,:); newCoord(2,:); newCoord(3,:)]'; % get rid of last dimension
        newcellTumor{i}.Patientisosurface=ones(size(finalCoord,1),3);
        newcellTumor{i}.Patientisosurface(:,1) = finalCoord(:,1);
        newcellTumor{i}.Patientisosurface(:,2) = finalCoord(:,2);
        newcellTumor{i}.Patientisosurface(:,3) = finalCoord(:,3);
    end
end


