function [newcellTumor] = convert_to_real_world(thetumor) 
%This script transforms the point cloud data in cell array for (our example is of Lung Adenocarcinoma Lesions) to
%real world coordinates.  This mat file loads a struct of size 41 lesions which
%includes contains the original data (var name: orgdata), the delianted data(var name: data), 
%slices where lesions exist in the orginal dicom (var name: slice) 
%slice thickness (var name: thickness), patient-barcode (var name: patientname),
%and dicom-info (var name: dcminfo)
%
%Returns transformation matrix real world coordinates

%load the header
inf = thetumor.allSliceDcminfo; % each slice has it's own header, but almost all fields are the same
if length(thetumor.slice) > 1
    nSl = thetumor.numslices; % number of ROI slices
else
    nSl = 1;
end


nY = double(inf{1}.Height);
nX = double(inf{1}.Width);
T1 = double(inf{1}.ImagePositionPatient); % position of first slice in the ROI

%load pixel spacing / scaling / resolution
RowColSpacing = double(inf{1}.PixelSpacing);

dx = double(RowColSpacing(1));
dX = [1; 1; 1].*dx;%cols
dy = double(RowColSpacing(2));
dY = [1; 1; 1].*dy;%rows
dz = double(inf{1}.SliceThickness); %thickness between slices
dZ = [1; 1; 1].*dz;

%directional cosines per basis vector
dircosXY = double(inf{1}.ImageOrientationPatient);
dircosX = dircosXY(1:3);
dircosY = dircosXY(4:6);

if nSl == 1
    dircosZ = cross(dircosX,dircosY);%orthogonal to other two direction cosines!
else
    %TN = double(-eval(['inf.PerFrameFunctionalGroupsSequence.Item_',sprintf('%d', N),'.PlanePositionSequence.Item_1.ImagePositionPatient']));
    % 'TN' is the ‘ImagePositionPatient’ vector for the last header in the list for this volume, if there is more than one header in the volume.
    TN = double(inf{nSl}.ImagePositionPatient); 
    dircosZ = ((T1-TN)./nSl)./dZ;
end

%all dircos together
dimensionmixing = [dircosX dircosY dircosZ];

%all spacing together
dimensionscaling = [dX dY dZ];

%mixing and spacing of dimensions together
R = dimensionmixing.*dimensionscaling;%maps from image basis to patientbasis

%offset and R together
A = [[R T1];[0 0 0 1]];

%you probably want to switch X and Y
%(depending on how you load your dicom into a matlab array)
Aold = A;
A(:,1) = Aold(:,2); % Y is the first row 
A(:,2) = Aold(:,1); % X is the second row

newcellTumor = A;
